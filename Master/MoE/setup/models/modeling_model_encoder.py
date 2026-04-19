import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from setup.utils.bucket_batch_sampler import BucketBatchSampler, collate_fn
from setup.utils.time_series_dataset import TimeSeriesDataset, load_jsonl
from setup.utils.early_stopping import EarlyStopping


from setup.experts.moirai_expert import MoiraiExpert
from setup.experts.timemoe_expert import TimeMoEExpert
from setup.experts.timesfm_expert import TimesFMExpert
from setup.experts.timer_expert import TimerExpert
from setup.experts.chronos_expert import ChronosExpert
from setup.utils.logging_train import get_log_dir_from_save_path, append_experts_weights, append_train_loss
from setup.utils.custom_loss_function import moe_custom_loss

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

EXPERT_CLASS_MAP = {
    "Moirai":    MoiraiExpert,
    "Time-MoE":  TimeMoEExpert,
    "TimesFM":   TimesFMExpert,
    "Timer":     TimerExpert,
    "Chronos":   ChronosExpert,
}

# -------------------
# Time Series Encoder
# -------------------
class TimeSeriesEncoder(nn.Module):
    """
    Multi-scale encoder for variable-length time series.
    Based on InceptionTime (Fawaz et al., 2020), arXiv:1909.04939.

    Solves the fixed-context-length problem:
        nn.Linear(context_length, N) tied the gating to one fixed length.
        This encoder replaces it with Conv1d(padding='same') + Global Average
        Pooling, fully removing the dependency on L.

    Flow (example B=32, L=168, hidden=32, out=64):
        (B, L)
        → unsqueeze(1)              (B, 1, L)
        → InstanceNorm1d            (B, 1, L)    per-sample normalisation
        → Bottleneck Conv k=1       (B, 32, L)   1 → hidden_dim channels
        → Branch k=3                (B, 32, L)   short-term patterns
        → Branch k=9                (B, 32, L)   medium-term patterns
        → Branch k=21               (B, 32, L)   long-term patterns
        → Skip MaxPool + Conv k=1   (B, 32, L)   peak detector
        → cat(dim=1)                (B, 128, L)  hidden_dim * 4 channels
        → mean(dim=-1)              (B, 128)     L collapses — length invariance
        → Linear(128→64) + GELU     (B, 64)
        → Dropout(0.1)
        → Linear(64→64)             (B, 64)      = out_dim (fixed)

    References:
        - InceptionTime: Fawaz et al. (2020), arXiv:1909.04939
        - GroupNorm: Wu & He (2018), arXiv:1803.08494
        - Global Average Pooling: Lin et al. (2013)
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        out_dim: int = 64,
        kernel_sizes: tuple = (3, 9, 21),
    ):
        super().__init__()

        # Per-instance normalisation — scale-invariant across different datasets
        # affine=True lets the model learn a post-norm rescale
        self.instance_norm = nn.InstanceNorm1d(1, affine=True)

        # 1×1 bottleneck: projects 1 input channel → hidden_dim channels
        # GroupNorm instead of BatchNorm to avoid NaN with batch_size=1
        self.bottleneck = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
            nn.GELU(),
        )

        # Parallel branches with different kernel sizes
        # padding='same' keeps temporal length L unchanged — key to length invariance
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding="same", bias=False),
                nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])

        # MaxPool skip branch: parameter-free peak detector
        self.skip_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
            nn.GELU(),
        )

        # MLP projection after Global Average Pooling
        total_channels = hidden_dim * (len(kernel_sizes) + 1)  # +1 for skip
        self.proj = nn.Sequential(
            nn.Linear(total_channels, out_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(out_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) or (B, 1, L) — L can vary between calls.
        Returns: (B, out_dim) — fixed size regardless of L.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)                                    # (B, 1, L)

        x = self.instance_norm(x)                                 # (B, 1, L)
        h = self.bottleneck(x)                                    # (B, hidden, L)

        branch_outs = [branch(h) for branch in self.branches]    # [(B, hidden, L) * n_kernels]
        skip_out = self.skip_branch(h)                            # (B, hidden, L)

        combined = torch.cat(branch_outs + [skip_out], dim=1)    # (B, hidden*4, L)
        pooled = combined.mean(dim=-1)                            # (B, hidden*4) — L collapses here

        return self.proj(pooled)                                  # (B, out_dim)


# -------------------
# MoERouter
# -------------------
class MoERouter(nn.Module):
    """
    Mixture-of-Experts router for time series forecasting.

    The gating network is now:
        series (B, L) → TimeSeriesEncoder → (B, encoder_out_dim) → MLP → (B, num_experts)

    This replaces the previous nn.Linear(context_length, num_experts) which
    required a fixed context length. The encoder handles any L.

    Experts remain frozen (zero-shot) — only encoder + gating are trained.
    """

    def __init__(
        self,
        encoder_hidden_dim: int = 32,
        encoder_out_dim: int = 64,
        kernel_sizes: tuple = (3, 9, 21),
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_out_dim = encoder_out_dim
        self.kernel_sizes = tuple(kernel_sizes)

        self.expert_keys = list(EXPERT_CLASS_MAP.keys())
        self.num_experts = len(self.expert_keys)

        # Frozen foundation model experts
        self.experts = nn.ModuleDict({
            k: EXPERT_CLASS_MAP[k](device=device)
            for k in self.expert_keys
        })

        # Variable-length encoder: (B, L) → (B, encoder_out_dim)
        self.encoder = TimeSeriesEncoder(
            hidden_dim=encoder_hidden_dim,
            out_dim=encoder_out_dim,
            kernel_sizes=kernel_sizes,
        )

        # Gating MLP: fixed embedding → expert logits
        self.gating = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim // 2),
            nn.GELU(),
            nn.Linear(encoder_out_dim // 2, self.num_experts),
        )

        # Noisy Top-K Gating (Switch Transformer, 2021): learned noise std
        self.noise_linear = nn.Linear(encoder_out_dim, self.num_experts)

        # Xavier init for gating layers
        for layer in self.gating:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Freeze experts: no grad, permanently in eval mode
        for ex in self.experts.values():
            for p in ex.parameters():
                p.requires_grad = False
            ex.eval()

        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        horizon: int,
        dir_csv_experts,
        top_k: int = 2,
        use_noise: bool = False,
        verbose: bool = False,
        sample_offset: int = 0,
    ):
        """
        x:               (B, L) — L can vary between calls.
        horizon:         number of future steps to forecast.
        dir_csv_experts: path for routing weight logs.
        top_k:           number of experts selected per sample.
        use_noise:       add Gaussian noise to logits during training.
        verbose:         print routing decisions.
        sample_offset:   offset for log indices when accumulating across batches.

        Returns: (final_preds, probs_clean, topk_idx)
            final_preds: (B, horizon)
            probs_clean: (B, num_experts) — softmax before noise/sparsity
            topk_idx:    (B, top_k)       — selected expert indices
        """
        x_device = x.to(self.device)
        batch_size = x_device.size(0)

        # 1. Encode variable-length series → fixed embedding
        context_repr = self.encoder(x_device)                    # (B, encoder_out_dim)

        # 2. Compute gating logits
        logits = self.gating(context_repr)                       # (B, E)
        probs_clean = F.softmax(logits, dim=-1)

        if use_noise and self.training:
            noise_std = F.softplus(self.noise_linear(context_repr))
            final_logits = logits + torch.randn_like(logits) * noise_std
        else:
            final_logits = logits

        # 3. Sparse top-k selection: set -inf on non-selected experts
        top_k_logits, topk_idx = torch.topk(final_logits, k=top_k, dim=-1)
        zeros = torch.full_like(final_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, topk_idx, top_k_logits)
        probs = F.softmax(sparse_logits, dim=-1)                 # (B, E)

        # 4. Log routing weights
        for i in range(topk_idx.size(0)):
            learners = [self.expert_keys[idx.item()] for idx in topk_idx[i]]
            weights = probs[i, topk_idx[i]].detach().cpu().numpy()
            append_experts_weights(
                dir_csv_experts,
                sample_idx=sample_offset + i,
                learners=learners,
                weights=weights,
            )

        # 5. Vectorized expert inference (one call per expert, on its sub-batch)
        preds_by_expert = torch.zeros(
            (self.num_experts, batch_size, horizon), device=self.device
        )

        for expert_idx in range(self.num_experts):
            mask = (topk_idx == expert_idx).any(dim=1)           # (B,)
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)

            if idxs.numel() == 0:
                continue

            xb_for_expert = x_device[idxs].clone()
            expert_module = self.experts[self.expert_keys[expert_idx]]

            with torch.no_grad():
                out = expert_module(xb_for_expert, prediction_length=horizon)

            preds_by_expert[expert_idx, idxs, :] = out.to(self.device).float().detach()

        # 6. Weighted combination (soft routing)
        # probs: (B, E), preds_by_expert: (E, B, H) → expert_preds: (B, E, H)
        expert_preds = preds_by_expert.permute(1, 0, 2)
        final_preds = torch.einsum("be,beh->bh", probs, expert_preds)  # (B, H)

        if verbose:
            for i in range(batch_size):
                chosen = topk_idx[i].tolist()
                sel = ", ".join(
                    f"{self.expert_keys[int(j)]}: {float(probs[i, j]):.3f}" for j in chosen
                )
                not_chosen = [j for j in range(self.num_experts) if j not in chosen]
                not_sel = ", ".join(
                    f"{self.expert_keys[j]}: {float(probs[i, j]):.3f}" for j in not_chosen
                )
                print(f"Sample {i}: Selected -> {sel}; Not selected -> {not_sel}")

        return final_preds, probs_clean, topk_idx

    def save(self, path: str):
        """
        Saves encoder, gating, and noise_linear states plus hyperparams
        needed to reconstruct the model on load.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "gating_state":       self.gating.state_dict(),
                "encoder_state":      self.encoder.state_dict(),
                "noise_state":        self.noise_linear.state_dict(),
                "expert_keys":        self.expert_keys,
                "num_experts":        self.num_experts,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "encoder_out_dim":    self.encoder_out_dim,
                "kernel_sizes":       list(self.kernel_sizes),
            },
            path,
        )
        print(f"Model saved in {path}")

    @staticmethod
    def load(path: str, device: str = "cpu") -> "MoERouter":
        """
        Reconstructs MoERouter from checkpoint, restoring encoder + gating states.
        No context_length required — the model is length-agnostic.
        """
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = MoERouter(
            encoder_hidden_dim=ckpt.get("encoder_hidden_dim", 32),
            encoder_out_dim=ckpt.get("encoder_out_dim", 64),
            kernel_sizes=tuple(ckpt.get("kernel_sizes", [3, 9, 21])),
            device=device,
        )
        model.gating.load_state_dict(ckpt["gating_state"])
        model.encoder.load_state_dict(ckpt["encoder_state"])
        model.noise_linear.load_state_dict(ckpt["noise_state"])
        model.to(device)
        model.eval()
        return model

# -------------------
# Train and Save
# -------------------
def train_and_save(
    data_path: str,
    horizon: int,
    save_path: str,
    use_noise,
    top_k: int = 2,
    norm: str = "std",
    device: str = "cpu",
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 1e-4,
    seed: int = 0,
):
    log_dir = get_log_dir_from_save_path(save_path)
    csv_experts = log_dir / "experts_weights_train.csv"

    use_noise_bool = (use_noise == "true") if isinstance(use_noise, str) else bool(use_noise)

    random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed_all(seed)

    sequences = load_jsonl(data_path)

    train_dataset = TimeSeriesDataset(sequences, horizon)

    train_sampler = BucketBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)

    model = MoERouter(device=device)
    model.to(device)

    # Train encoder + gating + noise_linear; experts are frozen
    trainable_params = (
        list(model.encoder.parameters())
        + list(model.gating.parameters())
    )
    opt = torch.optim.Adam(trainable_params, lr=lr)
    loss_fn = nn.HuberLoss(delta=2.0, reduction="mean")
    early_stopping = EarlyStopping(patience=10)

    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            if norm == "std":
                # -------------------------
                # Standard Scaler
                # -------------------------
                mean = data.mean(dim=1, keepdim=True)     
                std = data.std(dim=1, keepdim=True)       
                data_norm = (data - mean) / (std + 1e-8)
                target_norm = (target - mean) / (std + 1e-8)
            
            else:
                # -------------------------
                # Min-Max
                # -------------------------
                data_min = data.min(dim=1, keepdim=True).values
                data_max = data.max(dim=1, keepdim=True).values
                data_norm = (data - data_min) / (data_max - data_min + 1e-8)
                target_norm = (target - data_min) / (data_max - data_min + 1e-8)

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------
            output = model(data_norm, horizon=horizon, dir_csv_experts=csv_experts, use_noise=use_noise_bool, top_k=top_k)
            
            # -------------------------------------------------
            # Loss
            # -------------------------------------------------
            preds_norm, probs_clean, topk_idx = output

            loss = moe_custom_loss(
                preds=preds_norm,
                targets=target_norm,
                probs_clean=probs_clean,
                topk_idx=topk_idx,
                pred_loss_fn=loss_fn,
                alpha=0.02, 
            )

            # -------------------------------------------------
            # Backprop
            # -------------------------------------------------
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * data.size(0)
        
        train_loss /= len(train_loader.dataset)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.6f}")

        # Save Epoch | Train Loss
        csv_loss = log_dir / "train_loss.csv"
        append_train_loss(
            csv_loss,
            epoch=epoch + 1,
            train_loss=train_loss
        )

        early_stopping(train_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    early_stopping.load_best_model(model)

    # -------------------
    # Save
    # ------------------
    model.save(save_path)
    print(f'Saving model to {save_path}')

    return model

# -------------------
# Predict
# -------------------
def predict_from_model(
    model_path: str,
    series,
    horizon: int,
    top_k: int,
    use_noise,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Loads a saved MoERouter and performs prediction.
    The full series is used as context — no fixed context_length required.

    series: list/array/tensor of shape (T,) or (batch, T).
    Returns: tensor of shape (1, horizon) for 1D input, (batch, horizon) for 2D.
    """
    log_dir = get_log_dir_from_save_path(model_path)
    csv_experts = log_dir / "experts_weights_pred.csv"

    model = MoERouter.load(model_path, device=device)
    series = torch.as_tensor(series, dtype=torch.float32)
    use_noise_bool = (use_noise == "true") if isinstance(use_noise, str) else bool(use_noise)

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("`horizon` must be an int >= 1.")

    if series.dim() == 1:
        x = series.unsqueeze(0).to(device)            # (1, T)
        with torch.no_grad():
            out, _, _ = model(
                x=x,
                horizon=horizon,
                dir_csv_experts=csv_experts,
                top_k=top_k,
                use_noise=use_noise_bool,
                verbose=verbose,
            )
        return out.cpu()                              # (1, horizon)

    elif series.dim() == 2:
        x = series.to(device)                        # (batch, T)
        with torch.no_grad():
            out, _, _ = model(
                x=x,
                horizon=horizon,
                dir_csv_experts=csv_experts,
                top_k=top_k,
                use_noise=use_noise_bool,
                verbose=verbose,
            )
        return out.cpu()                              # (batch, horizon)

    else:
        raise ValueError(f"`series` must be 1D or 2D, got shape {tuple(series.shape)}")
