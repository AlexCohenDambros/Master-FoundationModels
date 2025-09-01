import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

from setup.experts.moirai_expert import MoiraiMoEExpert
from setup.experts.timemoe_expert import TimeMoEExpert
from setup.experts.timesfm_expert import TimesFMExpert

EXPERT_CLASS_MAP = {
    "moirai": MoiraiMoEExpert,
    "timemoe": TimeMoEExpert,
    "timesfm": TimesFMExpert,
}

# -------------------
# Dataset 
# -------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, context_length, horizon):
        self.samples = []
        for seq in sequences:
            if len(seq) >= context_length + horizon:
                inp = seq[:context_length]
                tgt = seq[context_length: context_length + horizon]
                self.samples.append((inp, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.tensor(inp, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)


# -------------------
# MoERouter
# -------------------
class MoERouter(nn.Module):
    def __init__(self, context_length, horizon, device="cpu"):
        super().__init__()
        self.device = device
        self.horizon = horizon
        self.context_length = context_length
        self.expert_keys = list(EXPERT_CLASS_MAP.keys())

        self.experts = nn.ModuleDict({
            k: EXPERT_CLASS_MAP[k](prediction_length=horizon, context_length=context_length, device=device)
            for k in self.expert_keys
        })

        # gating
        self.gating = nn.Linear(context_length, len(self.expert_keys))
        self.to(device)

    def forward(self, x, top_k: int = 2, verbose: bool = False):
        """
        x: (B, context_length) tensor (pode estar em CPU)
        Retorna: (B, horizon) tensor com predições combinadas.
        Importante: os experts são chamados com torch.no_grad() e usando clones dos inputs,
        evitando qualquer modificação in-place do tensor de entrada que quebraria o gradiente.
        """
        x_device = x.to(self.device)
        logits = self.gating(x_device)               # (B, n_experts)
        probs = F.softmax(logits, dim=-1)            # (B, n_experts)

        # top-k indices and values
        topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)  # shapes (B, k)
        B = x_device.size(0)
        final_preds = torch.zeros((B, self.horizon), device=self.device)

        # Para evitar chamar um expert B vezes (ineficiente) e também evitar in-place,
        # chamamos cada expert apenas nas amostras do batch que o selecionaram como parte do top-k.
        # Construímos um mapa expert -> list(indices) e executamos por expert uma vez.
        # Em seguida montamos as saídas e aplicamos os pesos renormalizados.
        # Primeiro inicializamos tensor para armazenar todas as predições por expert (desanexadas)
        # Note: vamos preencher a estrutura `expert_outs` com zeros e só escrever nas linhas correspondentes.

        n_experts = len(self.expert_keys)
        device = self.device

        # preds_by_expert: shape (n_experts, B, pred_len) , inicializado com zeros e tipo float32
        preds_by_expert = torch.zeros((n_experts, B, self.horizon), device=device)

        # Para cada expert, recolher os índices do batch onde ele está presente em topk_idx
        for expert_idx in range(n_experts):
            # encontrar onde expert_idx aparece em topk_idx (torch.eq and any over k)
            # mask shape (B,) boolean
            mask = (topk_idx == expert_idx).any(dim=1)  # True se o expert está entre top-k daquela amostra
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if idxs.numel() == 0:
                continue  # esse expert não selecionado para nenhuma amostra do batch
            # preparar entradas para este expert: clone para evitar in-place modifications propagarem
            xb_for_expert = x_device[idxs].clone()  # shape (m, context_length)
            # chamar expert em no_grad (experts são zero-shot)
            expert_key = self.expert_keys[expert_idx]
            expert_module = self.experts[expert_key]
            with torch.no_grad():
                out = expert_module(xb_for_expert)  # espera (m, horizon)
            out = out.to(device).float()
            # colocar nas posições corretas em preds_by_expert
            preds_by_expert[expert_idx, idxs, :] = out  # preenche só as linhas idxs

        # Agora combine apenas os top-k experts com pesos renormalizados por amostra
        for i in range(B):
            vals = topk_vals[i]                       # (k,)
            idxs = topk_idx[i]                        # (k,)
            # renormalizar entre os top-k
            s = vals.sum()
            if s <= 0:
                weights = vals.new_full(vals.shape, 1.0 / vals.shape[0])
            else:
                weights = vals / (s + 1e-8)
            # pegar os preds correspondentes (k, pred_len)
            chosen_preds = preds_by_expert[idxs, i, :]  # (k, pred_len)
            # multiplicar pesos (k,1) * (k, pred_len) e somar
            combined = (weights.unsqueeze(-1) * chosen_preds).sum(dim=0)  # (pred_len,)
            final_preds[i] = combined
            
            if verbose:
                selected_names = [self.expert_keys[idx] for idx in idxs]
                selected_str = ", ".join(f"{name}: {w:.3f}" for name, w in zip(selected_names, weights))
            
                not_selected_idx = [j for j in range(len(self.expert_keys)) if j not in idxs]
                not_selected_names = [self.expert_keys[j] for j in not_selected_idx]
                not_selected_weights = [probs[i, j].item() for j in not_selected_idx]
                not_selected_str = ", ".join(f"{name}: {w:.3f}" for name, w in zip(not_selected_names, not_selected_weights))
                
                print(f"Amostra {i}: Selecionados -> {selected_str}; Não selecionados -> {not_selected_str}")
        return final_preds

    def save(self, path):
        torch.save({
            "gating_state": self.gating.state_dict(),
            "expert_keys": self.expert_keys,
            "horizon": self.horizon,
            "context_length": self.context_length
        }, path)
        print(f"Modelo salvo em {path}")

    @staticmethod
    def load(path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        context_length = ckpt.get("context_length")
        horizon = ckpt["horizon"]
        model = MoERouter(context_length=context_length, horizon=horizon, device=device)
        model.gating.load_state_dict(ckpt["gating_state"])
        model.to(device)
        model.eval()
        return model


# -------------------
# Load Data
# -------------------
def load_jsonl(path):
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if not l:
                continue
            obj = json.loads(l)
            seq = obj.get("sequence")
            if seq is None:
                raise ValueError("Cada linha JSONL deve ter a chave 'sequence'")
            seqs.append([float(x) for x in seq])
    return seqs


def train_and_save(data_path, context_length, horizon, save_path, device="cpu",
                   batch_size=8, epochs=5, lr=1e-3, seed=0, detect_anomaly=False):
    """
    Treina apenas o gating (experts são zero-shot e permanecem congelados).
    """
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    random.seed(seed)
    torch.manual_seed(seed)

    sequences = load_jsonl(data_path)
    ds = TimeSeriesDataset(sequences, context_length, horizon)

    if len(ds) == 0:
        raise RuntimeError("Nenhuma amostra gerada. Verifique context_length/horizon e os dados.")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = MoERouter(context_length=context_length, horizon=horizon, device=device)
    # Otimizamos somente os parâmetros do gating
    opt = torch.optim.Adam(model.gating.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=2.0)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb  # permanece em CPU; MoERouter fará to(self.device) internamente para gating
            yb = yb.to(device)
            preds = model(xb)        # chama forward (internamente move gating inputs para device)
            loss = loss_fn(preds, yb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            count += 1
        print(f"Epoch {ep+1}/{epochs} - loss: {total_loss / max(1, count):.6f}")

    # salvar estado do gating e metadados
    model.save(save_path)
    print(f'Saving model to {save_path}')

    return model


def predict_from_model(model_path, series, context_length, device="cpu", verbose=True):
    model = MoERouter.load(model_path, device=device)
    if len(series) < model.context_length:
        raise ValueError(f"Série muito curta. context_length={model.context_length}")
    x = torch.tensor(series[-model.context_length:], dtype=torch.float32).unsqueeze(0)  # (1, context_length)
    with torch.no_grad():
        out = model(x, verbose=verbose)
    return out.cpu()