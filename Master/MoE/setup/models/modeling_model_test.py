import os
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

from setup.experts.moirai_expert import MoiraiExpert
from setup.experts.timemoe_expert import TimeMoEExpert
from setup.experts.timesfm_expert import TimesFMExpert
from setup.experts.timer_expert import TimerExpert
from setup.experts.chronos_expert import ChronosExpert
from setup.utils.logging_train import get_log_dir_from_save_path, append_experts_weights, append_train_loss
from setup.utils.custom_loss_function import moe_custom_loss

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EXPERT_CLASS_MAP = {
    "Moirai": MoiraiExpert,
    "Time-MoE": TimeMoEExpert,
    "TimesFM": TimesFMExpert,
    "Timer": TimerExpert,
    "Chronos": ChronosExpert,
}

# -------------------
# Dataset 
# -------------------
class TimeSeriesDataset(Dataset):
    # PT: Dataset de séries temporais onde:
    #     - inp = sequência antes do horizonte
    #     - tgt = últimos "horizon" pontos da sequência
    #
    # EN: Time series dataset where:
    #     - inp = sequence before the horizon
    #     - tgt = last "horizon" points of the sequence
    #
    # Example:
    # sequences = [[1,2,3,4,5,6,7,8]]
    # horizon = 2
    # sample -> inp=[1,2,3,4,5,6], tgt=[7,8]

    def __init__(self, sequences, horizon):
        self.samples = []

        for seq in sequences:
            if len(seq) > horizon:
                inp = seq[:-horizon]
                tgt = seq[-horizon:]
                self.samples.append((inp, tgt))

    # -------------------------------------------------------------------------
    # PT: Retorna o número de amostras disponíveis no dataset.
    #     Exemplo: len(dataset) -> 100
    # EN: Returns the number of samples available in the dataset.
    #     Example: len(dataset) -> 100
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # -------------------------------------------------------------------------
    # PT: Retorna a amostra (entrada, alvo) na posição idx em formato tensor.
    #     Exemplo: dataset[0] -> (tensor([1,2,3,4,5]), tensor([6,7]))
    # EN: Returns the (input, target) sample at position idx as tensors.
    #     Example: dataset[0] -> (tensor([1,2,3,4,5]), tensor([6,7]))
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.tensor(inp, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

# -------------------
# Encoder
# -------------------

class TimeSeriesEncoder(nn.Module):
    """
    PT:
    Encoder multi-escala para séries temporais de comprimento variável.
    Inspirado no InceptionTime (Fawaz et al., 2020).
 
    Problema resolvido:
        A camada nn.Linear(context_length, N) usada anteriormente no gating
        amarrava o modelo a um comprimento fixo. Este encoder substitui essa
        camada por convoluções 1D com padding='same' + Global Average Pooling,
        eliminando completamente a dependência do comprimento L.
 
    Fluxo resumido:
        (B, L)  →  (B, 1, L)  →  InstanceNorm
                →  Bottleneck Conv k=1  →  (B, hidden, L)
                →  [k=3 | k=9 | k=21 | MaxPool] em paralelo
                →  cat → (B, hidden*4, L)
                →  Global Avg Pool → (B, hidden*4)
                →  MLP → (B, out_dim)   ← tamanho FIXO, L desapareceu
 
    Referências:
        - InceptionTime: Fawaz et al. (2020), arXiv:1909.04939
        - GroupNorm: Wu & He (2018), arXiv:1803.08494
        - Global Average Pooling em classificação de séries: Lin et al. (2013)
 
    EN:
    Multi-scale encoder for variable-length time series.
    Inspired by InceptionTime (Fawaz et al., 2020).
 
    Problem solved:
        The nn.Linear(context_length, N) used previously in the gating tied the
        model to a fixed length. This encoder replaces that layer with Conv1d
        with padding='same' + Global Average Pooling, fully removing the
        dependency on length L.
 
    Flow summary:
        (B, L)  →  (B, 1, L)  →  InstanceNorm
                →  Bottleneck Conv k=1  →  (B, hidden, L)
                →  [k=3 | k=9 | k=21 | MaxPool] in parallel
                →  cat → (B, hidden*4, L)
                →  Global Avg Pool → (B, hidden*4)
                →  MLP → (B, out_dim)   ← FIXED size, L is gone
 
    References:
        - InceptionTime: Fawaz et al. (2020), arXiv:1909.04939
        - GroupNorm: Wu & He (2018), arXiv:1803.08494
        - Global Average Pooling for series classification: Lin et al. (2013)
    """
 
    def __init__(
        self,
        hidden_dim: int = 32,
        out_dim: int = 64,
        kernel_sizes: tuple = (3, 9, 21),
    ):
        super().__init__()
 
        # ------------------------------------------------------------------
        # PT: Normalização por instância.
        #     Normaliza cada série individualmente (média 0, std 1),
        #     independente das outras amostras no batch.
        #     Essencial quando séries vêm de domínios ou escalas diferentes
        #     (ex: energia em GWh vs temperatura em °C).
        #     affine=True permite que o modelo aprenda um rescale pós-norm.
        #
        # EN: Per-instance normalization.
        #     Normalizes each series individually (mean 0, std 1),
        #     independent of other samples in the batch.
        #     Critical when series come from different domains or scales
        #     (e.g., energy in GWh vs temperature in °C).
        #     affine=True lets the model learn a post-norm rescale.
        # ------------------------------------------------------------------
        self.instance_norm = nn.InstanceNorm1d(1, affine=True)
 
        # ------------------------------------------------------------------
        # PT: Bottleneck 1×1 — projeta de 1 canal para hidden_dim canais.
        #     Não olha para vizinhos temporais (kernel=1), apenas projeta
        #     cada ponto no tempo. Reduz custo das convoluções seguintes
        #     e aprende uma "paleta" de features base.
        #
        #     GroupNorm substituiu BatchNorm1d para evitar falha com
        #     batch size = 1, que causaria variância zero e NaN.
        #     num_groups=min(8, hidden_dim) garante que num_groups <= num_channels.
        #
        # EN: 1×1 bottleneck — projects from 1 channel to hidden_dim channels.
        #     Does not look at temporal neighbors (kernel=1), only projects
        #     each time point. Reduces cost of subsequent convolutions
        #     and learns a base "palette" of features.
        #
        #     GroupNorm replaced BatchNorm1d to avoid failure with
        #     batch size = 1, which would cause zero variance and NaN.
        #     num_groups=min(8, hidden_dim) ensures num_groups <= num_channels.
        # ------------------------------------------------------------------
        self.bottleneck = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
            nn.GELU(),
        )
 
        # ------------------------------------------------------------------
        # PT: Ramos paralelos — convoluções com kernels de tamanho diferente.
        #     padding='same' garante que a saída tenha o mesmo comprimento L
        #     da entrada, independente do kernel. Isso é o que mantém o
        #     encoder agnóstico ao comprimento da série.
        #
        #     Intuição dos kernels:
        #       k=3  → padrões de curto prazo (transições abruptas, picos)
        #       k=9  → padrões de médio prazo (ciclos diários se passo=1h)
        #       k=21 → padrões de longo prazo (tendências semanais)
        #
        #     Se suas séries têm sazonalidades muito longas, considere
        #     adicionar k=63 ou k=105.
        #
        # EN: Parallel branches — convolutions with different kernel sizes.
        #     padding='same' ensures output has the same length L as input,
        #     regardless of kernel size. This is what keeps the encoder
        #     agnostic to the series length.
        #
        #     Kernel intuition:
        #       k=3  → short-term patterns (abrupt transitions, spikes)
        #       k=9  → medium-term patterns (daily cycles if step=1h)
        #       k=21 → long-term patterns (weekly trends)
        #
        #     If your series have very long seasonalities, consider
        #     adding k=63 or k=105.
        # ------------------------------------------------------------------
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=k,
                    padding="same",
                    bias=False,
                ),
                nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])
 
        # ------------------------------------------------------------------
        # PT: Ramo de skip com MaxPool.
        #     MaxPool captura o valor máximo em cada janela de 3 passos —
        #     funciona como um detector de picos sem parâmetros aprendíveis.
        #     A Conv 1×1 seguinte projeta de volta para hidden_dim canais.
        #     Complementa os ramos conv que capturam padrões suaves.
        #
        # EN: MaxPool skip branch.
        #     MaxPool captures the maximum value in each 3-step window —
        #     works as a parameter-free peak detector.
        #     The following 1×1 Conv projects back to hidden_dim channels.
        #     Complements the conv branches that capture smooth patterns.
        # ------------------------------------------------------------------
        self.skip_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
            nn.GELU(),
        )
 
        # ------------------------------------------------------------------
        # PT: Projeção final após Global Average Pooling.
        #     total_channels = hidden_dim * (num_branches + 1 skip)
        #     O MLP reduz a dimensão e aprende combinações não-lineares
        #     das features multi-escala antes do gating.
        #
        # EN: Final projection after Global Average Pooling.
        #     total_channels = hidden_dim * (num_branches + 1 skip)
        #     The MLP reduces dimension and learns non-linear combinations
        #     of multi-scale features before gating.
        # ------------------------------------------------------------------
        total_channels = hidden_dim * (len(kernel_sizes) + 1)
        self.proj = nn.Sequential(
            nn.Linear(total_channels, out_dim),
            nn.GELU(),
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
        PT:
        Parâmetros:
            x: tensor de shape (B, L) ou (B, 1, L).
               L pode variar entre chamadas — o encoder é invariante ao comprimento.
 
        Retorna:
            Tensor de shape (B, out_dim) — tamanho fixo, independente de L.
 
        Fluxo de shapes com exemplo concreto (B=32, L=336, hidden=32, out=64):
            (32, 336)
            → unsqueeze(1)           : (32, 1, 336)
            → InstanceNorm1d         : (32, 1, 336)   normalizado por amostra
            → bottleneck Conv k=1    : (32, 32, 336)  1 → hidden_dim canais
            → branch k=3             : (32, 32, 336)  L preservado pelo padding='same'
            → branch k=9             : (32, 32, 336)
            → branch k=21            : (32, 32, 336)
            → skip MaxPool+Conv      : (32, 32, 336)
            → cat(dim=1)             : (32, 128, 336) hidden_dim * 4 canais
            → mean(dim=-1)           : (32, 128)      L colapsado — ponto chave
            → Linear(128→64) + GELU  : (32, 64)
            → Linear(64→64)          : (32, 64)       = out_dim
 
        EN:
        Parameters:
            x: tensor of shape (B, L) or (B, 1, L).
               L can vary between calls — the encoder is length-invariant.
 
        Returns:
            Tensor of shape (B, out_dim) — fixed size, independent of L.
        """
        # PT: Garante shape (B, 1, L) para Conv1d
        # EN: Ensure shape (B, 1, L) for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)                                   # (B, 1, L)
 
        # PT: Normalização por instância — cada série vira média≈0, std≈1
        # EN: Per-instance normalization — each series becomes mean≈0, std≈1
        x = self.instance_norm(x)                                # (B, 1, L)
 
        # PT: Bottleneck: projeta para hidden_dim canais
        # EN: Bottleneck: project to hidden_dim channels
        h = self.bottleneck(x)                                   # (B, hidden, L)
 
        # PT: Ramos paralelos: cada um recebe h e produz (B, hidden, L)
        # EN: Parallel branches: each receives h and produces (B, hidden, L)
        branch_outs = [branch(h) for branch in self.branches]   # list of (B, hidden, L)
        skip_out    = self.skip_branch(h)                        # (B, hidden, L)
 
        # PT: Concatena todos os ramos na dimensão de canais
        # EN: Concatenate all branches along the channel dimension
        combined = torch.cat(branch_outs + [skip_out], dim=1)   # (B, hidden*4, L)
 
        # PT: Global Average Pooling — colapsa L → escalar por canal.
        #     Esta operação elimina a dependência do comprimento da série.
        # EN: Global Average Pooling — collapses L → scalar per channel.
        #     This operation eliminates the series length dependency.
        pooled = combined.mean(dim=-1)                           # (B, hidden*4)
 
        # PT: Projeção MLP → representação final de tamanho fixo
        # EN: MLP projection → final fixed-size representation
        return self.proj(pooled)                                 # (B, out_dim)

# -------------------
# MoERouter
# -------------------
class MoERouter(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int = 32,
        encoder_out_dim: int = 64,
        kernel_sizes: tuple = (3, 9, 21),
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
 
        # PT: Keys dos experts na ordem definida
        # EN: Expert keys in defined order
        self.expert_keys = list(EXPERT_CLASS_MAP.keys())
        self.num_experts  = len(self.expert_keys)
 
        # PT: Instancia cada expert a partir do mapa
        # EN: Instantiate each expert from the map
        self.experts = nn.ModuleDict({
            k: EXPERT_CLASS_MAP[k](device=device)
            for k in self.expert_keys
        })
 
        # PT: Encoder multi-escala — extrai representação de tamanho fixo
        #     independente do comprimento L da série de entrada.
        # EN: Multi-scale encoder — extracts fixed-size representation
        #     regardless of the input series length L.
        self.encoder = TimeSeriesEncoder(
            hidden_dim=encoder_hidden_dim,
            out_dim=encoder_out_dim,
            kernel_sizes=kernel_sizes,
        )
 
        # PT: Gating MLP — mapeia representação fixa para logits sobre experts.
        #     Recebe encoder_out_dim (não mais context_length).
        # EN: Gating MLP — maps fixed representation to logits over experts.
        #     Receives encoder_out_dim (no longer context_length).
        self.gating = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim // 2),
            nn.GELU(),
            nn.Linear(encoder_out_dim // 2, self.num_experts),
        )
 
        # PT: Noise linear para Noisy Top-K Gating (Switch Transformer, 2021).
        #     Aprende o desvio padrão do ruído a adicionar aos logits durante
        #     treino, evitando que o modelo sempre escolha os mesmos experts
        #     (expert collapse). Corrigido: recebe encoder_out_dim.
        # EN: Noise linear for Noisy Top-K Gating (Switch Transformer, 2021).
        #     Learns the std of noise to add to logits during training,
        #     preventing the model from always choosing the same experts
        #     (expert collapse). Fixed: receives encoder_out_dim.
        self.noise_linear = nn.Linear(encoder_out_dim, self.num_experts)
 
        # PT: Inicialização Xavier para o gating
        # EN: Xavier initialization for gating
        for layer in self.gating:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
 
        nn.init.xavier_uniform_(self.noise_linear.weight)
        nn.init.zeros_(self.noise_linear.bias)
 
        # PT: Congela experts: sem grad, modo eval permanente.
        #     O gradiente flui apenas pelo encoder e pelo gating.
        # EN: Freeze experts: no grad, permanently in eval mode.
        #     Gradient flows only through encoder and gating.
        for ex in self.experts.values():
            for p in ex.parameters():
                p.requires_grad = False
            ex.eval()
 
        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        horizon: int,
        dir_csv_experts: str,
        top_k: int = 2,
        use_noise: bool = False,
        verbose: bool = False,
        sample_offset: int = 0,
    ) -> torch.Tensor:
        """
        PT:
        Parâmetros:
            x               : tensor (B, L) — L pode variar entre chamadas.
            horizon         : número de passos futuros a prever.
            dir_csv_experts : diretório para salvar os logs de roteamento.
            top_k           : número de experts selecionados por amostra.
            use_noise       : se True e em treino, adiciona ruído gaussiano
                              nos logits para evitar expert collapse.
            verbose         : se True, imprime informações de roteamento.
 
        Retorna:
            Tensor (B, H) com as predições combinadas dos experts selecionados.
 
        EN:
        Parameters:
            x               : tensor (B, L) — L can vary between calls.
            horizon         : number of future steps to forecast.
            dir_csv_experts : directory to save routing logs.
            top_k           : number of experts selected per sample.
            use_noise       : if True and in training, adds Gaussian noise
                              to logits to prevent expert collapse.
            verbose         : if True, prints routing information.
 
        Returns:
            Tensor (B, H) with the combined predictions from selected experts.
        """
        x_device = x.to(self.device)
        batch_size = x_device.size(0)
 
        # ------------------------------------------------------------------
        # PT: 1. Codifica a série para representação fixa (B, encoder_out_dim).
        #        Esta é a única mudança no forward em relação à versão anterior:
        #        context_repr substitui o x_device que ia direto pro gating.
        # EN: 1. Encode series to fixed representation (B, encoder_out_dim).
        #        This is the only forward change vs the previous version:
        #        context_repr replaces x_device going directly to gating.
        # ------------------------------------------------------------------
        context_repr = self.encoder(x_device)                    # (B, encoder_out_dim)
 
        # ------------------------------------------------------------------
        # PT: 2. Gating — computa logits e aplica ruído se necessário.
        # EN: 2. Gating — compute logits and apply noise if needed.
        # ------------------------------------------------------------------
        logits = self.gating(context_repr)                       # (B, E)

        probs_clean = F.softmax(logits, dim=-1)
 
        if use_noise and self.training:
            # PT: Ruído gaussiano com std aprendido — Noisy Top-K Gating.
            #     F.softplus garante que o std seja sempre positivo.
            # EN: Gaussian noise with learned std — Noisy Top-K Gating.
            #     F.softplus ensures std is always positive.
            noise_std    = F.softplus(self.noise_linear(context_repr))
            final_logits = logits + torch.randn_like(logits) * noise_std
        else:
            final_logits = logits
 
        # ------------------------------------------------------------------
        # PT: 3. Seleção top-k esparsa.
        #        Coloca -inf nos experts não selecionados antes do softmax,
        #        resultando em peso 0 para eles após a normalização.
        # EN: 3. Sparse top-k selection.
        #        Sets -inf on non-selected experts before softmax,
        #        resulting in weight 0 for them after normalization.
        # ------------------------------------------------------------------
        top_k_logits, topk_idx = torch.topk(final_logits, k=top_k, dim=-1)
 
        zeros         = torch.full_like(final_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, topk_idx, top_k_logits)
        probs         = F.softmax(sparse_logits, dim=-1)         # (B, E)
 
        if verbose:
            for i in range(batch_size):
                selected = [self.expert_keys[idx.item()] for idx in topk_idx[i]]
                weights  = probs[i, topk_idx[i]].detach().cpu().tolist()
                print(f"[MoERouter] sample {i}: {list(zip(selected, [f'{w:.3f}' for w in weights]))}")
 
        # ------------------------------------------------------------------
        # PT: 4. Log de roteamento (mantido do código original).
        # EN: 4. Routing log (kept from original code).
        # ------------------------------------------------------------------
        for i in range(topk_idx.size(0)):
            learners = [self.expert_keys[idx.item()] for idx in topk_idx[i]]
            weights  = probs[i, topk_idx[i]].detach().cpu().numpy()
            append_experts_weights(
                dir_csv_experts,
                sample_idx=sample_offset + i,  
                learners=learners,
                weights=weights,
            )
 
        # ------------------------------------------------------------------
        # PT: 5. Inferência dos experts selecionados.
        #        Para cada expert, coletamos as amostras do batch que o
        #        incluíram no top-k e chamamos o expert UMA VEZ com o
        #        sub-batch inteiro (vetorizado).
        #
        #        IMPORTANTE: os experts recebem x_device (série original),
        #        não context_repr (embedding). Eles são foundation models
        #        zero-shot que esperam séries temporais brutas.
        #
        # EN: 5. Inference of selected experts.
        #        For each expert, we collect the batch samples that included
        #        it in the top-k and call the expert ONCE with the entire
        #        sub-batch (vectorized).
        #
        #        IMPORTANT: experts receive x_device (original series),
        #        not context_repr (embedding). They are zero-shot foundation
        #        models that expect raw time series.
        # ------------------------------------------------------------------
        preds_by_expert = torch.zeros(
            (self.num_experts, batch_size, horizon), device=self.device
        )
 
        for expert_idx in range(self.num_experts):
            mask = (topk_idx == expert_idx).any(dim=1)           # (B,) bool
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
 
            if idxs.numel() == 0:
                continue
 
            xb_for_expert = x_device[idxs].clone()
            expert_key    = self.expert_keys[expert_idx]
            expert_module = self.experts[expert_key]
 
            with torch.no_grad():
                out = expert_module(
                    xb_for_expert,
                    prediction_length=horizon,
                )
 
            preds_by_expert[expert_idx, idxs, :] = (
                out.to(self.device).float().detach()
            )
 
        # ------------------------------------------------------------------
        # PT: 6. Combinação ponderada (soft routing).
        #        einsum "be,beh->bh":
        #          b = batch, e = experts, h = horizon
        #          Para cada amostra b, soma expert_preds[b,e,h] * probs[b,e]
        #          ao longo de e. Experts com peso 0 não contribuem.
        # EN: 6. Weighted combination (soft routing).
        #        einsum "be,beh->bh":
        #          b = batch, e = experts, h = horizon
        #          For each sample b, sums expert_preds[b,e,h] * probs[b,e]
        #          over e. Experts with weight 0 do not contribute.
        # ------------------------------------------------------------------
        expert_preds = preds_by_expert.permute(1, 0, 2)          # (B, E, H)
        final_preds  = torch.einsum("be,beh->bh", probs, expert_preds)  # (B, H)

        if verbose:
            for i in range(batch_size):

                idxs = topk_idx[i]

                chosen_list = idxs.tolist()
                selected_names = [self.expert_keys[int(j)] for j in chosen_list]
                selected_weights = probs[i, chosen_list].tolist()

                selected_str = ", ".join(
                    f"{name}: {float(w):.3f}"
                    for name, w in zip(selected_names, selected_weights)
                )

                not_selected_idx = [j for j in range(self.num_experts) if j not in chosen_list]
                not_selected_names = [self.expert_keys[j] for j in not_selected_idx]
                not_selected_weights = [float(probs[i, j].item()) for j in not_selected_idx]

                not_selected_str = ", ".join(
                    f"{name}: {w:.3f}"
                    for name, w in zip(not_selected_names, not_selected_weights)
                )

                print(
                    f"Sample {i}: "
                    f"Selected -> {selected_str}; "
                    f"Not selected -> {not_selected_str}"
                )

        return final_preds, probs_clean, topk_idx

    def save(self, path):
        """
        PT: Salva apenas o estado do gating e metadados necessários para reconstruir o roteador.
            OBS: não salvamos checkpoints dos experts grandes (assumimos que serão re-instanciados.
        EN: Saves only the gating state and metadata needed to rebuild the router.
            OBS: We do not save checkpoints for large experts (we assume they will be reinstantiated via EXPERT_CLASS_MAP after loading); 
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "gating_state": self.gating.state_dict(),
            "expert_keys": self.expert_keys,
            "num_experts": self.num_experts
        }, path)
        print(f"Model saved in {path}")

    @staticmethod
    def load(path, device="cpu"):
        """
        PT: Reconstrói MoERouter a partir do checkpoint. Requer que EXPERT_CLASS_MAP esteja disponível
            e que a mesma ordem de chaves seja usada.
        
        EN: Rebuilds the MoERouter from the checkpoint. Requires EXPERT_CLASS_MAP to be available
            and the same key order to be used.
        """
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = MoERouter(device=device)
        model.gating.load_state_dict(ckpt["gating_state"])
        model.to(device)
        model.eval()
        return model

# -------------------
# EarlyStopping
# -------------------
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, loss, model):
        score = -loss  
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

# -------------------
# Load Data
# -------------------
def load_jsonl(path):
    # -----------------------------------------------------------------------------
    # PT: Carrega um arquivo no formato JSONL (JSON por linha), extrai a chave
    #     "sequence" de cada linha e converte os valores em float.
    #     Exemplo de entrada: arquivo JSONL com linhas:
    #         {"sequence": [1, 2, 3]}
    #         {"sequence": [4, 5, 6]}
    #     Saída: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    #
    # EN: Loads a JSONL (JSON per line) file, extracts the "sequence" key from each
    #     line, and converts the values to float.
    #     Example input: JSONL file with lines:
    #         {"sequence": [1, 2, 3]}
    #         {"sequence": [4, 5, 6]}
    #     Output: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    # =============================================================================

    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if not l:
                continue
            obj = json.loads(l)
            seq = obj.get("sequence")
            if seq is None:
                raise ValueError("Each JSONL line must have the key 'sequence'")
            seqs.append([float(x) for x in seq])
    return seqs

def collate_fn_list(batch):
    """
    Retorna listas em vez de tensores empilhados.
    Cada série mantém seu comprimento original.
    """
    datas, targets = zip(*batch)
    return list(datas), list(targets)

# -------------------
# Train and Save Model
# ------------------
def train_and_save(data_path, horizon, save_path, use_noise, top_k=2, norm="std", device="cpu",
                   batch_size=32, epochs=20, lr=1e-4, seed=0):
    # =============================================================================
    # PT: Treina apenas o roteador (gating) do modelo MoERouter usando uma base de 
    #     séries temporais e salva o modelo treinado. Os experts permanecem 
    #     congelados (zero-shot). 
    #     O treinamento é feito em batches com função de perda Huber e um 
    #     regularizador de balanceamento simples para evitar colapso de roteamento.
    #
    #     - `data_path`: caminho para arquivo JSONL com as séries temporais
    #     - `horizon`: horizonte de previsão (número de passos futuros a prever)
    #     - `save_path`: caminho para salvar o modelo treinado (ex: "checkpoints/model.pt")
    #     - `top_k:` Número de especialistas a serem selecionados por amostra.
    #     - `device`: dispositivo ("cpu" ou "cuda")
    #     - `batch_size`: tamanho do lote para treino
    #     - `epochs`: número de épocas de treinamento
    #     - `lr`: taxa de aprendizado do otimizador
    #     - `seed`: semente aleatória para reprodutibilidade
    #     - `use_noise`: Se definido como True, aplicará ruído durante o treinamento do modelo.
    #
    #     Saída: modelo MoERouter treinado (instância do objeto)
    #
    # EN: Trains only the router (gating) of the MoERouter model using a dataset of 
    #     time series and saves the trained model. The experts remain frozen 
    #     (zero-shot). 
    #     Training is performed in batches with Huber loss and a simple 
    #     load-balancing regularizer to avoid routing collapse.
    #
    #     - `data_path`: path to JSONL file with time series
    #     - `horizon`: forecast horizon (number of future steps to predict)
    #     - `save_path`: path to save the trained model (e.g., "checkpoints/model.pt")
    #     - `top_k:` number of experts to pick per sample.
    #     - `device`: device ("cpu" or "cuda")
    #     - `batch_size`: training batch size
    #     - `epochs`: number of training epochs
    #     - `lr`: learning rate for the optimizer
    #     - `seed`: random seed for reproducibility
    #     - `use_noise`: If set to True, it will apply noise during model training.
    #
    #     Output: trained MoERouter model (object instance)
    # =============================================================================

    #===============================================
    #   STARTING TRAINING
    #===============================================
    log_dir = get_log_dir_from_save_path(save_path)
    csv_experts = log_dir / "experts_weights_train.csv"

    use_noise = True if use_noise=="true" else False

    random.seed(seed)
    torch.manual_seed(seed)
    
    if "cuda" in device:
        torch.cuda.manual_seed_all(seed)

    ds = load_jsonl(data_path)

    train_dataset = TimeSeriesDataset(ds, horizon)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,   # 32
        shuffle=False,
        collate_fn=collate_fn_list,
    )

    model = MoERouter(device=device)
    model.to(device)

    opt = torch.optim.Adam(model.gating.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=2.0, reduction='mean')

    early_stopping = EarlyStopping(patience=10)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for datas, targets in train_loader:
            opt.zero_grad()
            batch_loss = 0
            sample_counter = 0

            for data, target in zip(datas, targets):
                # (L,) → (1, L)
                data   = data.unsqueeze(0).to(device)
                target = target.unsqueeze(0).to(device)

                if norm == "std":
                    mean = data.mean(dim=1, keepdim=True)
                    std  = data.std(dim=1, keepdim=True).clamp(min=1e-8)
                    data_norm   = (data - mean) / std
                    target_norm = (target - mean) / std
                else:
                    data_min = data.min(dim=1, keepdim=True).values
                    data_max = data.max(dim=1, keepdim=True).values
                    data_norm   = (data - data_min) / (data_max - data_min + 1e-8)
                    target_norm = (target - data_min) / (data_max - data_min + 1e-8)

                output = model(data_norm, horizon=horizon, dir_csv_experts=csv_experts, use_noise=use_noise, top_k=top_k, sample_offset=sample_counter)
                sample_counter += 1

                preds_norm, probs_clean, topk_idx = output

                loss = moe_custom_loss(
                    preds=preds_norm,
                    targets=target_norm,
                    probs_clean=probs_clean,
                    topk_idx=topk_idx,
                    pred_loss_fn=loss_fn,
                    alpha=0.02,
                )


                (loss / len(datas)).backward()
                batch_loss += loss.item()

            opt.step()
            train_loss += batch_loss
        
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

    return model, early_stopping.best_score
# -------------------
# Predict model
# ------------------
def predict_from_model(model_path, series, horizon, top_k, use_noise, device="cpu", verbose=True):
    # =============================================================================
    # PT: Carrega um modelo MoERouter salvo e realiza previsão para uma ou mais
    #     séries temporais.
    #     - `series`: tensor 1D (T) ou 2D (batch, T)
    #     - `horizon`: número de passos a prever
    #     A entrada do modelo é toda a série disponível.
    #
    # EN: Loads a saved MoERouter model and performs prediction for one or more
    #     time series.
    #     - `series`: 1D tensor (T) or 2D (batch, T)
    #     - `horizon`: number of forecast steps
    #     The model receives the full series as input.
    # =============================================================================

    log_dir = get_log_dir_from_save_path(model_path)
    csv_experts = log_dir / "experts_weights_pred.csv"

    model = MoERouter.load(model_path, device=device)

    series = torch.as_tensor(series, dtype=torch.float32, device=device)

    use_noise = True if use_noise == "true" else False

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("`horizon` must be an int >= 1.")

    if series.dim() == 1:
        x = series.unsqueeze(0) 

        with torch.no_grad():
            out = model(
                x=x,
                horizon=horizon,
                dir_csv_experts=csv_experts,
                top_k=top_k,
                use_noise=use_noise,
                verbose=verbose
            )

        return out[0].cpu() 

    elif series.dim() == 2:
        x = series 

        with torch.no_grad():
            out = model(
                x=x,
                horizon=horizon,
                dir_csv_experts=csv_experts,
                top_k=top_k,
                use_noise=use_noise,
                verbose=verbose
            )

        return out.cpu() 

    else:
        raise ValueError(f"`series` must be 1D or 2D, but got shape {tuple(series.shape)}")
