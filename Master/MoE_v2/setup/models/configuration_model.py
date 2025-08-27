#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
configuration_model.py

Versão adaptada do MoE router para usar os experts fornecidos no seu projeto:
 - Usa EXPERT_CLASS_MAP para buscar classes: "moirai", "timemoe", "timesfm".
 - Separa main() (entrada/CLI) de configuration_model(args) (treino, avaliação, salvamento).
 - Gating treinado supervisionado por rótulos "melhor expert" obtidos via HuberLoss(delta=2.0).
 - Roteador top-k (default 2) -> softmax logits -> topk -> renormaliza -> média ponderada.
 - Salva checkpoint (gating state_dict + metadados) e salva figura 'plot_actual_vs_predicted.png'.
"""

import argparse
import json
import math
import os
import random
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


from ..experts.moirai_expert import MoiraiMoEExpert  # type: ignore
from ..experts.timemoe_expert import TimeMoEExpert  # type: ignore
from ..experts.timesfm_expert import TimesFMExpert  # type: ignore
EXPERT_CLASS_MAP = {
    "moirai": MoiraiMoEExpert,
    "timemoe": TimeMoEExpert,
    "timesfm": TimesFMExpert,
}
  
# -------------------------
# DummyExpert (caso import real não exista) — útil para demo/testes
# -------------------------
class DummyExpert(nn.Module):
    def __init__(self, input_length: int, prediction_length: int, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.input_length = input_length
        self.prediction_length = prediction_length

        # pequeno MLP para gerar predições de teste
        self.net = nn.Sequential(
            nn.Linear(input_length, 128),
            nn.ReLU(),
            nn.Linear(128, prediction_length)
        )
        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_length)
        x = x.to(self.device).float()
        return self.net(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

# -------------------------
# Dataset: JSONL -> janelas (sliding windows)
# -------------------------
class TimeSeriesWindowDataset(Dataset):
    def __init__(self, sequences: List[List[float]], input_length: int, prediction_length: int,
                 stride: int = 1, max_windows_per_series: int = None):
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.samples = []
        for seq in sequences:
            L = len(seq)
            max_start = L - (input_length + prediction_length)
            if max_start < 0:
                continue
            starts = range(0, max_start + 1, stride)
            if max_windows_per_series is not None:
                starts = list(starts)[:max_windows_per_series]
            for s in starts:
                inp = seq[s: s + input_length]
                tgt = seq[s + input_length: s + input_length + prediction_length]
                self.samples.append((inp, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.tensor(inp, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

# -------------------------
# MoE Router (gating + experts)
# -------------------------
class MoETimeRouter:
    def __init__(self, experts: List[nn.Module], input_length: int, prediction_length: int, top_k: int = 2, device: str = "cpu"):
        self.device = device
        self.experts = experts
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.n_experts = len(experts)
        self.top_k = max(1, min(top_k, self.n_experts))

        # gating MLP
        self.gating = nn.Sequential(
            nn.Linear(input_length, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_experts)
        ).to(self.device)
        self.gating.eval()

    def experts_predict(self, x: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        # x: (N, input_length)
        N = x.shape[0]
        preds = torch.zeros((N, self.n_experts, self.prediction_length), device=self.device)
        for k, expert in enumerate(self.experts):
            outs = []
            for i in range(0, N, batch_size):
                xb = x[i:i+batch_size].to(self.device)
                with torch.no_grad():
                    out = expert(xb)  # expect (B, pred_len)
                if out.ndim == 1:
                    out = out.unsqueeze(1)
                out = out.to(self.device).float()
                outs.append(out)
            out_full = torch.cat(outs, dim=0)
            preds[:, k, :] = out_full
        return preds

    def gating_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).float()
        return self.gating(x)

    def predict_weighted(self, x: torch.Tensor, top_k: int = None, verbose: bool = False) -> torch.Tensor:
        if top_k is None:
            top_k = self.top_k
        logits = self.gating_logits(x)  # (N, n_experts)
        probs = F.softmax(logits, dim=-1)  # (N, n_experts)
        preds = self.experts_predict(x)  # (N, n_experts, pred_len)
        N = x.shape[0]
        final = torch.zeros((N, self.prediction_length), device=self.device)
        topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)
        topk_sum = topk_vals.sum(dim=-1, keepdim=True)
        topk_sum[topk_sum == 0] = 1.0
        topk_weights = topk_vals / topk_sum
        for i in range(N):
            idxs = topk_idx[i]
            w = topk_weights[i]
            chosen = preds[i, idxs, :]  # (top_k, pred_len)
            combined = (w.unsqueeze(-1) * chosen).sum(dim=0)
            final[i] = combined
            if verbose:
                print(f"Amostra {i}: experts escolhidos {idxs.cpu().tolist()} pesos {w.cpu().tolist()}")
        return final

    def train_gating(self, dataloader_label: DataLoader, epochs: int = 5, lr: float = 1e-3, batch_size: int = 64,
                     device: str = "cpu", verbose: bool = True):
        device = self.device = device
        self.gating.to(device)
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.gating.parameters(), lr=lr)
        huber = nn.HuberLoss(reduction='none', delta=2.0)

        # Build labeled set: run experts on dataloader_label and compute best expert per sample
        Xs = []
        Ys = []
        print("Gerando rótulos para o gating (rodando todos os experts nas amostras de rotulagem)...")
        for xb, yb in dataloader_label:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                preds = self.experts_predict(xb)  # (B, n_experts, pred_len)
            B = preds.shape[0]
            losses = torch.zeros((B, self.n_experts), device=device)
            for k in range(self.n_experts):
                pk = preds[:, k, :]  # (B, pred_len)
                l = huber(pk, yb)  # (B, pred_len)
                l = l.mean(dim=1)  # (B,)
                losses[:, k] = l
            best_idx = torch.argmin(losses, dim=1)  # (B,)
            Xs.append(xb.cpu())
            Ys.append(best_idx.cpu())

        if len(Xs) == 0:
            raise RuntimeError("Nenhuma amostra rotulada (verifique input_length/pred_length e tamanhos).")
        X_label = torch.cat(Xs, dim=0)
        y_label = torch.cat(Ys, dim=0)
        label_ds = torch.utils.data.TensorDataset(X_label, y_label)
        label_loader = DataLoader(label_ds, batch_size=batch_size, shuffle=True)

        # Treino do gating
        for ep in range(epochs):
            self.gating.train()
            running = 0.0
            total = 0
            correct = 0
            for xb, lbl in label_loader:
                xb = xb.to(device)
                lbl = lbl.to(device)
                logits = self.gating(xb)
                loss = crit(logits, lbl)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == lbl).sum().item()
                total += xb.size(0)
            avg_loss = running / total
            acc = correct / total
            if verbose:
                print(f"[Epoch {ep+1}/{epochs}] loss={avg_loss:.4f} acc={acc:.4f}")
        self.gating.eval()

    def save_checkpoint(self, path: str, expert_keys: List[str], extra: dict = None):
        ckpt = {
            "gating_state_dict": self.gating.state_dict(),
            "input_length": self.input_length,
            "prediction_length": self.prediction_length,
            "top_k": self.top_k,
            "expert_keys": expert_keys,
            "extra": extra or {}
        }
        torch.save(ckpt, path)
        print(f"Checkpoint salvo em: {path}")

    @staticmethod
    def load_from_checkpoint(path: str, instantiate_fn, device: str = "cpu"):
        ckpt = torch.load(path, map_location='cpu')
        input_length = ckpt["input_length"]
        prediction_length = ckpt["prediction_length"]
        top_k = ckpt["top_k"]
        expert_keys = ckpt["expert_keys"]
        experts = []
        for k in expert_keys:
            exp = instantiate_fn(k, input_length, prediction_length, device=device)
            experts.append(exp)
        router = MoETimeRouter(experts=experts, input_length=input_length, prediction_length=prediction_length, top_k=top_k, device=device)
        router.gating.load_state_dict(ckpt["gating_state_dict"])
        router.gating.to(device)
        router.gating.eval()
        return router

# -------------------------
# Helpers: instantiate expert from key (uses EXPERT_CLASS_MAP)
# -------------------------
def instantiate_expert_from_key(key: str, input_length: int, prediction_length: int, device: str = "cpu"):
    key = key.strip().lower()
    if key in EXPERT_CLASS_MAP:
        cls = EXPERT_CLASS_MAP[key]
        try:
            # instanciar com assinatura (prediction_length, device)
            inst = cls(prediction_length=prediction_length, device=device)
        except Exception:
            try:
                inst = cls(input_length, prediction_length, device)
            except Exception as e:
                print(f"Erro ao instanciar expert '{key}': {e}. Usando DummyExpert.")
                inst = DummyExpert(input_length, prediction_length, device=device)
    else:
        print(f"Aviso: chave de expert '{key}' não encontrada em EXPERT_CLASS_MAP. Usando DummyExpert.")
        inst = DummyExpert(input_length, prediction_length, device=device)
    # se for nn.Module, move para device e coloca em eval()
    if isinstance(inst, nn.Module):
        inst.to(device)
        inst.eval()
    return inst

# -------------------------
# JSONL loader
# -------------------------
def load_jsonl_sequences(path: str):
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            seq = obj.get("sequence") or obj.get("values") or obj.get("series")
            if seq is None:
                raise ValueError("Cada linha JSONL deve conter a chave 'sequence'.")
            seqs.append([float(x) for x in seq])
    return seqs

# -------------------------
# configuration_model(args): faz todo o fluxo de treino/avaliação/salvamento
# -------------------------
def configuration_model(args):
    # Carrega sequências
    sequences = load_jsonl_sequences(args.data)
    print(f"Sequências carregadas: {len(sequences)}")

    # Monta dataset de janelas
    ds = TimeSeriesWindowDataset(sequences, input_length=args.input_length, prediction_length=args.pred_length, stride=args.stride)
    if len(ds) == 0:
        raise RuntimeError("Nenhuma janela gerada: verifique input_length/pred_length e tamanhos das séries.")
    print(f"Amostras (janelas) geradas: {len(ds)}")

    # Split simples (embaralhado)
    idxs = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(idxs)
    split = int((1.0 - args.test_size) * len(idxs))
    train_idxs = idxs[:split]
    test_idxs = idxs[split:]

    # definir subset para rotulagem do gating (última fração do treino)
    label_count = int(len(train_idxs) * args.gating_label_frac)
    if label_count < 1:
        label_count = min( max(1, int(len(train_idxs) * 0.2)), len(train_idxs) )
    gating_label_idxs = train_idxs[:label_count]
    gating_train_idxs = train_idxs[label_count:]  # não usado neste fluxo, mas mantido

    train_loader_for_label = DataLoader(torch.utils.data.Subset(ds, gating_label_idxs), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(ds, test_idxs), batch_size=args.batch_size, shuffle=False)

    expert_keys = list(EXPERT_CLASS_MAP.keys())
    experts = []
    for k in expert_keys:
        print(f"Instanciando expert '{k}' ...")
        e = instantiate_expert_from_key(k, args.input_length, args.pred_length, device=args.device)
        experts.append(e)

    # Build router e treinar gating
    router = MoETimeRouter(experts=experts, input_length=args.input_length, prediction_length=args.pred_length, top_k=args.top_k, device=args.device)
    router.train_gating(dataloader_label=train_loader_for_label, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=args.device, verbose=True)

    # Avaliação no test set (gera preds e calcula métricas)
    print("Avaliação no test set ...")
    all_preds = []
    all_tgts = []
    for xb, yb in test_loader:
        xb = xb.to(args.device)
        with torch.no_grad():
            preds = router.predict_weighted(xb, top_k=args.top_k, verbose=False)
        all_preds.append(preds.cpu())
        all_tgts.append(yb)
    if len(all_preds) == 0:
        raise RuntimeError("Conjunto de teste vazio - verifique tamanho do dataset/test split.")
    all_preds = torch.cat(all_preds, dim=0)
    all_tgts = torch.cat(all_tgts, dim=0)
    mse = F.mse_loss(all_preds, all_tgts).item()
    mae = F.l1_loss(all_preds, all_tgts).item()
    print(f"Test MSE: {mse:.6f}  MAE: {mae:.6f}")

    # Salva checkpoint (gating + metadados)
    router.save_checkpoint(args.save_path, expert_keys, extra={"mse": mse, "mae": mae})

    # Salva gráfico actual vs predicted (plota os primeiros N pontos do teste, achatando horizonte)
    try:
        # Flatten: compare valor por valor (apenas para visual)
        # Para plot simples, pegamos as primeiras 200 previsões (horizonte concatenado)
        max_plot = min(200, all_preds.shape[0] * all_preds.shape[1])
        # flatten targets and preds
        preds_flat = all_preds.detach().cpu().numpy().reshape(-1)[:max_plot]
        tgts_flat = all_tgts.detach().cpu().numpy().reshape(-1)[:max_plot]
        plt.figure(figsize=(10,4))
        plt.plot(tgts_flat, label="actual")
        plt.plot(preds_flat, label="predicted")
        plt.legend()
        plt.title("Actual vs Predicted (flattened horizon)")
        plt.xlabel("index")
        plt.ylabel("value")
        plt.tight_layout()
        plt.savefig("plot_actual_vs_predicted.png")
        plt.close()
        print("Gráfico salvo em: plot_actual_vs_predicted.png")
    except Exception as e:
        print("Não foi possível salvar gráfico:", e)

    print("Treinamento e salvamento concluídos.")

# -------------------------
# Funções de predição (usa checkpoint salvo)
# -------------------------
def predict_from_checkpoint(args):
    # Carrega router via função estática utilizando a função de instanciação por chave
    router = MoETimeRouter.load_from_checkpoint(args.load_path, instantiate_expert_from_key, device=args.device)

    # Se recebeu uma série via CLI
    if args.series is not None and len(args.series) > 0:
        series = [float(x) for x in args.series]
        if len(series) < router.input_length:
            raise ValueError(f"Série fornecida menor que input_length ({router.input_length})")
        inp = torch.tensor(series[-router.input_length:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = router.predict_weighted(inp.to(args.device), top_k=args.top_k, verbose=args.verbose)
        print("Predição (tensor):", out.cpu())
        return

    # Se recebeu JSONL
    if args.jsonl is not None:
        seqs = load_jsonl_sequences(args.jsonl)
        outs = []
        for seq in seqs:
            if len(seq) < router.input_length:
                print("Skipping seq menor que input_length:", len(seq))
                continue
            inp = torch.tensor(seq[-router.input_length:], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = router.predict_weighted(inp.to(args.device), top_k=args.top_k, verbose=args.verbose)
            outs.append(out.cpu().squeeze(0))
        if len(outs) > 0:
            stacked = torch.stack(outs, dim=0)
            print("Batch predictions shape:", stacked.shape)
            print(stacked.numpy())
        else:
            print("Nenhuma predição gerada.")
        return

    raise ValueError("Para predição informe --series ou --jsonl")

# -------------------------
# Demo helper
# -------------------------
def demo_mode(args):
    # cria sequências sintéticas, instância DummyExperts e executa fluxo curto
    print("Rodando demo com DummyExperts...")
    sequences = []
    for i in range(60):
        L = args.input_length + args.pred_length + 10
        seq = [math.sin((t + i) / 24.0) * 10.0 + 50.0 + random.random() for t in range(L)]
        sequences.append(seq)
    ds = TimeSeriesWindowDataset(sequences, args.input_length, args.pred_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    # experts dummy
    e1 = DummyExpert(args.input_length, args.pred_length, device=args.device)
    e2 = DummyExpert(args.input_length, args.pred_length, device=args.device)
    e3 = DummyExpert(args.input_length, args.pred_length, device=args.device)
    experts = [e1, e2, e3]
    router = MoETimeRouter(experts, args.input_length, args.pred_length, top_k=args.top_k, device=args.device)
    # usar metade para rotulagem
    n = len(ds)
    label_idx = list(range(n//2))
    label_loader = DataLoader(torch.utils.data.Subset(ds, label_idx), batch_size=args.batch_size, shuffle=False)
    router.train_gating(label_loader, epochs=args.epochs, device=args.device, verbose=True)
    # avaliar restante
    test_loader = DataLoader(torch.utils.data.Subset(ds, list(range(n//2, n))), batch_size=args.batch_size, shuffle=False)
    all_p, all_t = [], []
    for xb, yb in test_loader:
        with torch.no_grad():
            p = router.predict_weighted(xb.to(args.device), top_k=args.top_k, verbose=False)
        all_p.append(p.cpu())
        all_t.append(yb)
    all_p = torch.cat(all_p, dim=0)
    all_t = torch.cat(all_t, dim=0)
    print("Demo MSE:", F.mse_loss(all_p, all_t).item())