import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

import logging

from setup.experts.moirai_expert import MoiraiExpert
from setup.experts.moiraimoe_expert import MoiraiMoEExpert
from setup.experts.timemoe_expert import TimeMoEExpert
from setup.experts.timesfm_expert import TimesFMExpert
from setup.experts.timer_expert import TimerExpert
from setup.experts.chronos_expert import ChronosExpert

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

EXPERT_CLASS_MAP = {
    "Moirai": MoiraiExpert,
    # "Moirai-MoE": MoiraiMoEExpert,
    "Time-MoE": TimeMoEExpert,
    "TimesFM": TimesFMExpert,
    "Timer": TimerExpert,
    "Chronos": ChronosExpert,
}

logging.basicConfig(
    filename="log_training.txt",     
    filemode="a",                     
    format="%(asctime)s - %(message)s",
    level=logging.INFO
)

# -------------------
# Dataset 
# -------------------
class TimeSeriesDataset(Dataset):
    # PT: Representa um dataset de séries temporais, onde cada amostra é composta
    #     por uma janela de contexto (entrada) e um horizonte de previsão (saída).
    #     Exemplo de entrada: sequences = [[1,2,3,4,5,6,7,8]]
    #     context_length = 5, horizon = 2
    #     Amostra resultante: (inp=[1,2,3,4,5], tgt=[6,7])
    #
    # EN: Represents a time series dataset, where each sample consists of
    #     a context window (input) and a forecast horizon (output).
    #     Example input: sequences = [[1,2,3,4,5,6,7,8]]
    #     context_length = 5, horizon = 2
    #     Resulting sample: (inp=[1,2,3,4,5], tgt=[6,7])
    # =============================================================================

    # -------------------------------------------------------------------------
    # PT: Inicializa o dataset, cortando as sequências em pares (entrada, alvo).
    #     Entrada: sequences (lista de listas), context_length=5, horizon=2
    #     Saída: lista de amostras [(inp, tgt), ...]
    #
    # EN: Initializes the dataset, slicing sequences into (input, target) pairs.
    #     Input: sequences (list of lists), context_length=5, horizon=2
    #     Output: list of samples [(inp, tgt), ...]
    # -------------------------------------------------------------------------
    def __init__(self, sequences, context_length, horizon):
        self.samples = []
        for seq in sequences:
            if len(seq) >= context_length + horizon:
                inp = seq[:context_length]
                tgt = seq[context_length: context_length + horizon]
                self.samples.append((inp, tgt))

    # -------------------------------------------------------------------------
    # PT: Retorna o número de amostras disponíveis no dataset.
    #     Exemplo: len(dataset) -> 100
    #
    # EN: Returns the number of samples available in the dataset.
    #     Example: len(dataset) -> 100
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # -------------------------------------------------------------------------
    # PT: Retorna a amostra (entrada, alvo) na posição idx em formato tensor.
    #     Exemplo: dataset[0] -> (tensor([1,2,3,4,5]), tensor([6,7]))
    #
    # EN: Returns the (input, target) sample at position idx as tensors.
    #     Example: dataset[0] -> (tensor([1,2,3,4,5]), tensor([6,7]))
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.tensor(inp, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

# -------------------
# MoERouter
# -------------------
class MoERouter(nn.Module):
    """
    Router do Mixture-of-Experts (MoE) para séries temporais.
    Seleciona top-k experts usando noisy top-k gating + máscara esparsa,
    chama experts em no_grad (zero-shot) e combina predições ponderadas.
    """

    def __init__(self, context_length: int, device="cpu"):
        super().__init__()
        self.device = device

        # keys dos experts na ordem definida
        self.expert_keys = list(EXPERT_CLASS_MAP.keys())
        self.num_experts = len(self.expert_keys)

        # instancia experts
        self.experts = nn.ModuleDict({
            k: EXPERT_CLASS_MAP[k](device=device)
            for k in self.expert_keys
        })

        # gating + noise linear
        self.gating = nn.Linear(context_length, self.num_experts)
        self.noise_linear = nn.Linear(context_length, self.num_experts)

        # Inicializações mais adequadas (quebram simetria)
        nn.init.xavier_uniform_(self.gating.weight)
        nn.init.zeros_(self.gating.bias)

        nn.init.xavier_uniform_(self.noise_linear.weight)
        nn.init.zeros_(self.noise_linear.bias)
        # opcional: iniciar bias do noise negativo para ruído inicial menor
        # nn.init.constant_(self.noise_linear.bias, -2.0)

        # Congela experts (zero-shot)
        for ex in self.experts.values():
            for p in ex.parameters():
                p.requires_grad = False
            ex.eval()

        # Vars para balance loss / debugging
        self.last_probs = None
        self.last_topk_idx = None

        # Move para device por último
        self.to(device)

    def forward(self, x: torch.Tensor, context_length: int, horizon: int, top_k: int = 2, verbose: bool = False):
        device = self.device
        x_device = x.to(device)

        # logits base
        logits = self.gating(x_device)  # (batch_size, E)

        # === Noisy gating: adiciona ruído APENAS no treino ===
        if self.training:
            noise = self.noise_linear(x_device)
            noise_std = F.softplus(noise) 
            noisy_logits = logits + torch.randn_like(logits) * noise_std
        else:
            noisy_logits = logits

        # === Top-k sobre noisy_logits (esparsificação dos logits) ===
        topk_vals, topk_idx = torch.topk(noisy_logits, k=top_k, dim=-1)  # valores e índices dos top-k logits

        # cria logits esparsos: mantém apenas top-k (resto = -inf)
        mask = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = mask.scatter(-1, topk_idx, topk_vals)

        probs = F.softmax(sparse_logits, dim=-1)  # (batch_size, E) - só top-k terão >0

        # Salva para cálculo do balance loss / debug (sem grad)
        self.last_probs = probs.detach()
        self.last_topk_idx = topk_idx.detach()

        # >>>>>>> LOG <<<<<<<
        logging.info("\n=== Selected experts and weights per sample ===")
        for i in range(topk_idx.size(0)):
            chosen_experts = [self.expert_keys[idx.item()] for idx in topk_idx[i]]
            # usar as probabilidades normalizadas (probs), não os logits crus
            chosen_weights = probs[i, topk_idx[i]].detach().cpu().numpy()

            logging.info(f"Sample {i}:")
            for exp, w in zip(chosen_experts, chosen_weights):
                logging.info(f"   Expert: {exp} | Weight: {w:.4f}")
        logging.info("===============================================\n")
        # >>>>>>> END LOG <<<<<<<

        batch_size = x_device.size(0)
        final_preds = torch.zeros((batch_size, horizon), device=device)

        # acumula predições por expert: (num_experts, batch_size, horizon)
        preds_by_expert = torch.zeros((self.num_experts, batch_size, horizon), device=device)

        # chama cada expert apenas com o sub-batch relevante
        for expert_idx in range(self.num_experts):
            mask_bool = (topk_idx == expert_idx).any(dim=1)   # (batch_size,)
            idxs = torch.nonzero(mask_bool, as_tuple=False).squeeze(1)
            if idxs.numel() == 0:
                continue

            xb_for_expert = x_device[idxs].clone()
            expert_key = self.expert_keys[expert_idx]
            expert_module = self.experts[expert_key]

            with torch.no_grad():
                expert_name = expert_module.__class__.__name__
                if expert_name in ["TimeMoEExpert"]:
                    data_min = xb_for_expert.min(dim=1, keepdim=True).values
                    data_max = xb_for_expert.max(dim=1, keepdim=True).values
                    data_range = (data_max - data_min) + 1e-8
                    xb_norm = (xb_for_expert - data_min) / data_range
                    out_norm = expert_module(xb_norm, context_length=context_length, prediction_length=horizon)
                    out = out_norm * data_range + data_min
                else:
                    mean = xb_for_expert.mean(dim=1, keepdim=True)
                    std = xb_for_expert.std(dim=1, keepdim=True)
                    xb_norm = (xb_for_expert - mean) / (std + 1e-8)
                    out_norm = expert_module(xb_norm, context_length=context_length, prediction_length=horizon)
                    out = out_norm * (std + 1e-8) + mean

            out = out.to(device).float().detach()
            preds_by_expert[expert_idx, idxs, :] = out

        # combina predições usando as probabilidades normalizadas (probs)
        for i in range(batch_size):
            idxs = topk_idx[i]                     # (k,)
            weights = probs[i, idxs]               # (k,) já normalizado
            chosen_preds = preds_by_expert[idxs, i, :]   # (k, horizon)
            combined = (weights.unsqueeze(-1) * chosen_preds).sum(dim=0)
            final_preds[i] = combined

            # Printing which models were selected on the router
            if verbose:
                chosen_list = idxs.tolist()
                selected_names = [self.expert_keys[int(j)] for j in chosen_list]
                selected_str = ", ".join(
                    f"{name}: {float(w):.3f}" for name, w in zip(selected_names, weights.tolist())
                )

                not_selected_idx = [j for j in range(self.num_experts) if j not in chosen_list]
                not_selected_names = [self.expert_keys[j] for j in not_selected_idx]
                not_selected_weights = [float(probs[i, j].item()) for j in not_selected_idx]
                not_selected_str = ", ".join(f"{name}: {w:.3f}" for name, w in zip(not_selected_names, not_selected_weights))

                print(f"Sample: Selected -> {selected_str}; Not selected -> {not_selected_str}")

        return final_preds

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
    def load(path, context_length, device="cpu"):
        """
        PT: Reconstrói MoERouter a partir do checkpoint. Requer que EXPERT_CLASS_MAP esteja disponível
            e que a mesma ordem de chaves seja usada.
        
        EN: Rebuilds the MoERouter from the checkpoint. Requires EXPERT_CLASS_MAP to be available
            and the same key order to be used.
        """
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = MoERouter(context_length=context_length, device=device)
        model.gating.load_state_dict(ckpt["gating_state"])
        model.to(device)
        model.eval()
        return model

# -------------------
# EarlyStopping
# -------------------
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
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

# -------------------
# Train and Save Model
# ------------------
def train_and_save(data_path, context_length, horizon, save_path, device="cpu",
                   batch_size=32, epochs=30, lr=1e-3, seed=0, detect_anomaly=False):
    # =============================================================================
    # PT: Treina apenas o roteador (gating) do modelo MoERouter usando uma base de 
    #     séries temporais e salva o modelo treinado. Os experts permanecem 
    #     congelados (zero-shot). 
    #     O treinamento é feito em batches com função de perda Huber e um 
    #     regularizador de balanceamento simples para evitar colapso de roteamento.
    #
    #     - `data_path`: caminho para arquivo JSONL com as séries temporais
    #     - `context_length`: número de pontos de entrada usados como contexto
    #     - `horizon`: horizonte de previsão (número de passos futuros a prever)
    #     - `save_path`: caminho para salvar o modelo treinado (ex: "checkpoints/model.pt")
    #     - `device`: dispositivo ("cpu" ou "cuda")
    #     - `batch_size`: tamanho do lote para treino
    #     - `epochs`: número de épocas de treinamento
    #     - `lr`: taxa de aprendizado do otimizador
    #     - `seed`: semente aleatória para reprodutibilidade
    #     - `detect_anomaly`: ativa debug de gradientes (mais lento, útil para depuração)
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
    #     - `context_length`: number of input points used as context
    #     - `horizon`: forecast horizon (number of future steps to predict)
    #     - `save_path`: path to save the trained model (e.g., "checkpoints/model.pt")
    #     - `device`: device ("cpu" or "cuda")
    #     - `batch_size`: training batch size
    #     - `epochs`: number of training epochs
    #     - `lr`: learning rate for the optimizer
    #     - `seed`: random seed for reproducibility
    #     - `detect_anomaly`: enables gradient anomaly detection (slower, debug only)
    #
    #     Output: trained MoERouter model (object instance)
    # =============================================================================

    logging.info("===============================================")
    logging.info(f"STARTING TRAINING")
    logging.info(f"Model will be saved at: {save_path}")
    logging.info("===============================================")

    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    random.seed(seed)
    torch.manual_seed(seed)
    
    if "cuda" in device:
        torch.cuda.manual_seed_all(seed)

    ds = load_jsonl(data_path)

    train_dataset = TimeSeriesDataset(ds, context_length, horizon)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    model = MoERouter(context_length=context_length, device=device)
    model.to(device)

    opt = torch.optim.Adam(
        list(model.gating.parameters()) + list(model.noise_linear.parameters()),
        lr=lr
    )

    loss_fn = nn.HuberLoss(delta=2.0, reduction='mean')

    early_stopping = EarlyStopping(patience=3, delta=0.01)

    for epoch in range(epochs):

        model.train()

        train_loss = 0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            
            preds = model(data, context_length=context_length, horizon=horizon)

            loss = loss_fn(preds, target)

            alpha = 0.01  # ajuste: 0.001 .. 0.05
            mean_probs = model.last_probs.mean(dim=0)       # (E,)
            # fraction: fração de roteamentos por expert no batch (contagem top-k)
            counts = torch.zeros(model.num_experts, device=device)
            # model.last_topk_idx shape (batch_size, k)
            for idx in model.last_topk_idx.view(-1):
                counts[idx] += 1
            mean_fraction = counts / counts.sum()           # (E,)

            balance_loss = model.num_experts * torch.sum(mean_probs * mean_fraction)
            total_loss = loss + alpha * balance_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            train_loss += total_loss.item() * data.size(0)
        
        train_loss /= len(train_loader.dataset)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

        logging.info("-----------------------------------------------")
        logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
        logging.info("-----------------------------------------------")

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
# Predict model
# ------------------
def predict_from_model(model_path, series, context_length, horizon, device="cpu", verbose=True):
    # =============================================================================
    # PT: Carrega um modelo salvo do tipo MoERouter e realiza a previsão para uma
    #     ou várias séries temporais fornecidas. A série é cortada para o tamanho
    #     do contexto e passada ao modelo junto com o horizonte de previsão.
    #     - `model_path`: caminho do modelo salvo (ex: "checkpoints/model.pt")
    #     - `series`: tensor 1D (ex: torch.Size([462])) ou 2D (ex: torch.Size([8, 398]))
    #     - `context_length`: número de pontos usados como contexto (ex: 5)
    #     - `horizon`: número de passos a serem previstos (ex: 2)
    #     Saída: tensor 2D com previsões (ex: torch.Size([1, horizon]) ou [batch, horizon])
    #
    # EN: Loads a saved MoERouter model and performs prediction for one or more
    #     time series. Each series is trimmed to the context length and passed
    #     to the model along with the forecast horizon.
    #     - `model_path`: path to saved model (e.g., "checkpoints/model.pt")
    #     - `series`: 1D tensor (e.g., torch.Size([462])) or 2D (e.g., torch.Size([8, 398]))
    #     - `context_length`: number of points used as context (e.g., 5)
    #     - `horizon`: number of steps to forecast (e.g., 2)
    #     Output: 2D tensor with predictions (ex: torch.Size([1, horizon]) or [batch, horizon])
    # =============================================================================

    model = MoERouter.load(model_path, context_length=context_length, device=device)

    series = torch.as_tensor(series, dtype=torch.float32)

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("`horizon` must be an int >= 1.")

    # Case 1D
    if series.dim() == 1:
        if len(series) < context_length:
            raise ValueError("Series too short for the requested context")
        x = series[-context_length:].unsqueeze(0)  # (1, context_length)
        with torch.no_grad():
            out = model(x=x, context_length=context_length, horizon=horizon, verbose=verbose)
        return out.cpu()  # (1, horizon)

    # Case 2D
    elif series.dim() == 2:
        outs = []
        for row in series:
            if len(row) < context_length:
                raise ValueError("One of the series is too short for the requested context")
            x = row[-context_length:].unsqueeze(0)  # (1, context_length)
            with torch.no_grad():
                out = model(x=x, context_length=context_length, horizon=horizon, verbose=verbose)
            outs.append(out.cpu())
        return torch.cat(outs, dim=0)  # (batch, horizon)

    else:
        raise ValueError(f"`series` must be 1D or 2D, but got shape {tuple(series.shape)}")

# Note: !!!!!!Importante!!!!!!
# Ponto crítico, se o modelo for treinado com um context_lenght de 168 por exemplo, quando carregar e fazer uma previsao, o context_lenght deve ser igual o do treino.
# Todo: ponto de pequisa futura, verificar como deixar o context_lenght dinamico ou tentar implementar em intervalos.