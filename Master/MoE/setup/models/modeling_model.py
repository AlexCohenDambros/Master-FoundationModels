import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

from setup.experts.moirai_expert import MoiraiExpert
from setup.experts.moiraimoe_expert import MoiraiMoEExpert
from setup.experts.timemoe_expert import TimeMoEExpert
from setup.experts.timesfm_expert import TimesFMExpert
from setup.experts.timer_expert import TimerExpert
from setup.experts.chronos_expert import ChronosExpert

EXPERT_CLASS_MAP = {
    "Moirai": MoiraiExpert,
    # "Moirai-MoE": MoiraiMoEExpert,
    "Time-MoE": TimeMoEExpert,
    "TimesFM": TimesFMExpert,
    "Timer": TimerExpert,
    "Chronos": ChronosExpert,
}

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
    # =============================================================================
    # PT: Roteador (router) do Mixture-of-Experts para séries temporais.
    #     - Mantém um dicionário de experts (foundation models) instanciados via
    #       EXPERT_CLASS_MAP.
    #     - O gating (roteador) é uma camada linear que mapeia o contexto para
    #       logits sobre os experts. Seleciona top-k experts por amostra, chama
    #       cada expert em no_grad (zero-shot) e combina predições por pesos.
    #
    # EN: Router for Mixture-of-Experts for time-series.
    #     - Keeps a ModuleDict of experts created from EXPERT_CLASS_MAP.
    #     - The gating is a linear layer mapping context to logits over experts.
    #     - Selects top-k experts per sample, calls experts in no_grad (zero-shot)
    #       and combines predictions by weights.
    # =============================================================================

    def __init__(self, context_length:int, device="cpu"):
        super().__init__()
        self.device = device

        # PT: keys dos experts na ordem definida (lista de strings)
        # EN: keys of experts in the defined order (list of strings)
        self.expert_keys = list(EXPERT_CLASS_MAP.keys())

        # PT: número de experts
        # EN: number of experts
        self.num_experts = len(self.expert_keys)

        # PT: Instancia cada expert a partir do mapa (device)
        # EN: Instantiates each expert from the map (device)
        self.experts = nn.ModuleDict({
            k: EXPERT_CLASS_MAP[k](device=device)
            for k in self.expert_keys
        })

        # PT: Gating: mapeia vetor de contexto (tamanho context_length) para logits sobre experts
        # EN: Gating: maps context vector (size context_length) to logits over experts
        self.gating = nn.Linear(context_length, self.num_experts)

        # PT: Congelar os experts: desativa grad e coloca em eval(). Isso evita alocação de grad acidental dos experts e garante comportamento determinístico.
        # EN: Freeze experts: disables grad and places it in eval(). This prevents accidental grad allocation from experts and ensures deterministic behavior.
        for ex in self.experts.values():
            for p in ex.parameters():
                p.requires_grad = False
            ex.eval()  

        # PT: Move parâmetros/arquitetura para o dispositivo desejado
        # EN: Move parameters/architecture to the desired device
        self.to(device)

        # PT: Variáveis usadas para cálculo do balance-loss (monitor / regularização)
        # EN: Variables used to calculate balance-loss (monitor/regularization)
        # self.last_logits = None
        # self.last_topk_idx = None
        # self.last_topk_vals = None

    def forward(self, x: torch.Tensor, context_length: int, horizon: int, top_k: int = 2, verbose: bool = False):
        """
        PT:
        - x: tensor (batch_size, context_length) contendo o contexto por amostra.
        - top_k: número de experts a selecionar por amostra.
        Retorna: tensor (batch_size, horizon) com predições combinadas.

        EN:
        - x: tensor (batch_size, context_length) with context per sample.
        - top_k: number of experts to pick per sample.
        Returns: (batch_size, horizon) combined predictions.
        """

        # PT: move o input para o device do modelo (pode já estar no mesmo device; .to faz nada nesse caso)
        # EN: move the input to the model's device (it may already be on the same device; .to does nothing in that case)
        x_device = x.to(self.device)

        # computa logits (batch_size, E) e probs softmax (batch_size, E)
        logits = self.gating(x_device)                # raw scores do roteador / router raw scores
        probs = F.softmax(logits, dim=-1)             # probabilidades por expert (por amostra) / probabilities per expert (per sample) 

        # PT: pega top-k probabilidades e seus índices por amostra -> shapes (batch_size, k)
        # EN: get top-k probabilities and their indices per sample -> shapes (batch_size, k)
        topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)

        # >>>>>>> LOG/PRINT <<<<<<<
        # Show selected experts and their weights for each sample
        print("\n=== Selected experts and weights per sample ===")
        for i in range(topk_idx.size(0)):                     
            chosen_experts = [self.expert_keys[idx.item()] for idx in topk_idx[i]]
            chosen_weights = topk_vals[i].detach().cpu().numpy()
            print(f"Sample {i}:")
            for exp, w in zip(chosen_experts, chosen_weights):
                print(f"   Expert: {exp} | Weight: {w:.4f}")
        print("===============================================\n")
        # >>>>>>> END LOG <<<<<<<

        batch_size = x_device.size(0)  # batch_size

        # PT: tensor final que irá armazenar as predições combinadas (batch_size, horizon)
        # EN: final tensor that will store the combined predictions (batch_size, horizon)
        final_preds = torch.zeros((batch_size, horizon), device=self.device)

        # PT: salvar as informações relevantes para possível cálculo de regularizador (balance loss). usamos detach() para não manter grafo e não aumentar uso de memória
        # EN: save relevant information for possible regularizer calculation (balance loss). we use detach() to avoid maintaining the graph and not increase memory usage
        # self.last_logits = logits.detach()
        # self.last_topk_idx = topk_idx.detach()
        # self.last_topk_vals = topk_vals.detach()

        device = self.device

        '''
        PT: 
            - preds_by_expert: acumulador de predições por expert:
            - formato: (num_experts, batch_size, horizon)
            - inicializa com zeros; vamos preencher apenas as linhas correspondentes às amostras que escolheram o expert
        
        EN:
            - preds_by_expert: Accumulator of predictions by expert:
            - format: (num_experts, batch_size, horizon)
            - initializes with zeros; we will fill only the lines corresponding to the samples that chose the expert
        '''
        preds_by_expert = torch.zeros((self.num_experts, batch_size, horizon), device=device)

        # PT: Para cada expert, coletamos as amostras do batch que o incluíram no top-k; chamamos o expert **uma vez** com o sub-batch (vetorizado) — evita chamar expert N vezes.
        # EN: For each expert, we collect the samples from the batch that included it in the top-k; we call the expert **once** with the (vectorized) sub-batch — avoid calling expert N times.
        for expert_idx in range(self.num_experts):

            # PT: mask: booleano shape (batch_size,) indicando quais amostras possuem esse expert entre seus top-k
            # EN: mask: boolean shape (batch_size,) indicating which samples have this expert among their top-k
            mask = (topk_idx == expert_idx).any(dim=1)
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)

            if idxs.numel() == 0:
                continue

            # PT: extrai o sub-batch que será passado ao expert
            # EN: extracts the sub-batch that will be passed to the expert
            xb_for_expert = x_device[idxs].clone()  

            expert_key = self.expert_keys[expert_idx]
            expert_module = self.experts[expert_key]

            # PT: chamar expert em modo no_grad (zero-shot, sem computar gradientes)
            # EN: call expert in no_grad mode (zero-shot, no gradients computed)
            with torch.no_grad():
                out = expert_module(xb_for_expert, context_length=context_length, prediction_length=horizon)

            out = out.to(device).float().detach()
            preds_by_expert[expert_idx, idxs, :] = out

        # PT: Agora combinamos as predições *apenas* entre os top-k escolhidos para cada amostra. A implementação abaixo faz isso amostra a amostra (pode ser vetorizada para performance).
        # EN: Now we combine predictions *only* from the top-k chosen for each sample. The implementation below does this sample by sample (can be vectorized for performance).
        for i in range(batch_size):
            vals = topk_vals[i]   # (k,)
            idxs = topk_idx[i]    # (k,) indices dos experts escolhidos para a amostra i / (k,) indices of the experts chosen for sample i

            # PT: renormaliza pesos entre os top-k (evita soma 0)
            # EN: renormalizes weights among top-k (avoids sum to 0)
            s = vals.sum()
            if s <= 0:
                weights = vals.new_full(vals.shape, 1.0 / vals.shape[0])
            else:
                weights = vals / (s + 1e-8)

            # PT: extrai as predições correspondentes a esses experts para a amostra i: preds_by_expert[idxs, i, :] -> (k, horizon)
            # EN: extracts the predictions corresponding to these experts for sample i: preds_by_expert[idxs, i, :] -> (k, horizon)
            chosen_preds = preds_by_expert[idxs, i, :]

            # PT: aplica os pesos e soma para obter predição final (horizon,)
            # EN: apply the weights and sum to obtain the final prediction (horizon,)
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
        ckpt = torch.load(path, map_location=device)
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
# Load Data (with train/test split)
# -------------------
def load_jsonl2(path, test_ratio=0.2, seed=42):
    # -----------------------------------------------------------------------------
    # PT: Carrega um arquivo no formato JSONL contendo séries temporais com chaves
    #     variadas, por exemplo:
    #         {"etanolhidratado_pe": [31059.872, 22527.896, 23241.738]}
    #         {"gasolinadeaviacao_sp": [174.962, 174.962, 151.817]}
    #         {"oleocombustivel_pe": [18251.00816, 13803.07041]}
    #
    #     - Se state = None → separa treino e teste de forma aleatória
    #       (80% treino, 20% teste).
    #     - Se state = "sp" → todas as séries terminadas em "_sp" vão para teste,
    #       e o restante fica em treino.
    #
    #     Retorna: train_dataset, val_dataset
    #
    # EN: Loads a JSONL file containing time series with varied keys, e.g.:
    #         {"etanolhidratado_pe": [31059.872, 22527.896, 23241.738]}
    #         {"gasolinadeaviacao_sp": [174.962, 174.962, 151.817]}
    #         {"oleocombustivel_pe": [18251.00816, 13803.07041]}
    #
    #     - If state = None → splits into train and test randomly
    #       (80% train, 20% test).
    #     - If state = "sp" → all series ending with "_sp" go into test,
    #       and the rest go into train.
    #
    #     Returns: train_dataset, val_dataset
    # =============================================================================

    random.seed(seed)  

    series = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if not l:
                continue
            obj = json.loads(l)

            if len(obj) != 1:
                raise ValueError("Each line must contain exactly one series")

            name, values = list(obj.items())[0]
            seq = [float(x) for x in values]
            series.append((name, seq))

    train_dataset, val_dataset = [], []

    random.shuffle(series)
    split_idx = int(len(series) * (1 - test_ratio))
    train_dataset = [seq for _, seq in series[:split_idx]]
    val_dataset = [seq for _, seq in series[split_idx:]]

    return train_dataset, val_dataset

# -------------------
# Train and Save Model
# ------------------
def train_and_save(data_path, context_length, horizon, save_path, device="cpu",
                   batch_size=32, epochs=5, lr=1e-3, balance_coef=1e-2, seed=0, detect_anomaly=False):
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
    #     - `balance_coef`: coeficiente de regularização para balanceamento do roteador
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
    #     - `balance_coef`: regularization coefficient for router load balancing
    #     - `seed`: random seed for reproducibility
    #     - `detect_anomaly`: enables gradient anomaly detection (slower, debug only)
    #
    #     Output: trained MoERouter model (object instance)
    # =============================================================================

    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed_all(seed)

    # train_dataset, val_dataset = load_jsonl(data_path)
    ds = load_jsonl(data_path)

    train_dataset = TimeSeriesDataset(ds, context_length, horizon)
    # vaL_dataset = TimeSeriesDataset(val_dataset, context_length, horizon)

    # for name, dataset in [("train", train_dataset), ("val", val_dataset)]:
    #     if len(dataset) == 0:
    #         raise RuntimeError(f"No samples generated in {name} dataset. "
    #                            f"Check context_length, horizon, and your data.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # val_loader = DataLoader(vaL_dataset, batch_size=batch_size, shuffle=False)

    model = MoERouter(context_length=context_length, device=device)
    model.to(device)

    opt = torch.optim.Adam(model.gating.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=2.0, reduction='mean')

    early_stopping = EarlyStopping(patience=3, delta=0.01)

    # TODO: Add MinMax normalization to the data before passing it to experts 
    for epoch in range(epochs):

        model.train()

        train_loss = 0
        val_loss = 0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            # -------------------------
            # Normalização (Standard Scaler)
            # -------------------------
            # mean = data.mean(dim=1, keepdim=True)     # média por série [32,1]
            # std = data.std(dim=1, keepdim=True)       # desvio padrão por série [32,1]
            # data_norm = (data - mean) / (std + 1e-8)  # evitar divisão por zero
            # preds_norm = model(data_norm, context_length=context_length, horizon=horizon)
            # Desnormaliza previsões
            # preds = preds_norm * (std + 1e-8) + mean

            # -------------------------
            # Normalização (Min-Max)
            # -------------------------
            data_min = data.min(dim=1, keepdim=True).values
            data_max = data.max(dim=1, keepdim=True).values
            data_range = (data_max - data_min) + 1e-8  # evitar divisão por zero

            data_norm_minmax = (data - data_min) / data_range  # escala para [0,1]

            preds_norm_minmax = model(data_norm_minmax, context_length=context_length, horizon=horizon)

            preds = preds_norm_minmax * data_range + data_min
            
            # -------------------------
            # Normal
            # -------------------------
            # preds = model(data, context_length=context_length, horizon=horizon) 
            # loss = loss_fn(preds, target)

            loss = loss_fn(preds, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * data.size(0)
        
        train_loss /= len(train_loader.dataset)

        # model.eval()

        # with torch.no_grad():
        #     for data, target in val_loader:

        #         data = data.to(device)
        #         target = target.to(device)

        #         preds = model(data, context_length=context_length, horizon=horizon)

        #         loss = loss_fn(preds, target)
        #         val_loss += loss.item() * data.size(0)

        # val_loss /= len(val_loader.dataset)

        # print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

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