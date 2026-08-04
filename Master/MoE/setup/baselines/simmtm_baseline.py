# =============================================================================
# PT: Wrapper do SimMTM (thuml/SimMTM, NeurIPS 2023) para forecasting univariado.
#     SimMTM NÃO tem forecast zero-shot: é um framework de pré-treino + fine-tune.
#     Aqui usamos os pesos pré-treinados (se disponíveis) e fazemos fine-tune por
#     caso, no mesmo espírito dos modelos treinados na hora do pipeline
#     (PatchTST/NBEATS/RF/XGB). Cada série é tratada como univariada
#     (channel-independent). Devolve predições no ESPAÇO CRU (o modelo normaliza
#     internamente, estilo Non-stationary Transformer), como PatchTST.
#
# EN: SimMTM wrapper for univariate forecasting. SimMTM has no zero-shot forecast:
#     it is a pretrain + finetune framework. We load pretrained weights (if present)
#     and finetune per case, like the other train-on-the-fly models in the pipeline.
#     Each series is univariate (channel-independent). Returns predictions in RAW
#     space (the model normalizes internally), like PatchTST.
# =============================================================================
import os
import numpy as np
import torch
import torch.nn as nn

from ._vendor import import_model_class, SIMMTM_ROOT, SIMMTM_CKPT

# ---------------------------------------------------------------------------
# Hiperparâmetros (com override por variável de ambiente para os spikes/validação)
# ---------------------------------------------------------------------------
_MAX_SEQ_LEN = int(os.environ.get("SIMMTM_SEQ_LEN", "96"))   # janela de entrada L
_EPOCHS      = int(os.environ.get("SIMMTM_EPOCHS", "10"))
_BATCH_SIZE  = int(os.environ.get("SIMMTM_BATCH", "64"))
_LR          = float(os.environ.get("SIMMTM_LR", "1e-4"))

# Arquitetura (deve casar com o checkpoint pré-treinado do SimMTM que for usado;
# padrão estilo ETT: d_model=32, n_heads=16, d_ff=64, e_layers=3).
_D_MODEL   = int(os.environ.get("SIMMTM_DMODEL", "32"))
_N_HEADS   = int(os.environ.get("SIMMTM_NHEADS", "16"))
_D_FF      = int(os.environ.get("SIMMTM_DFF", "64"))
_E_LAYERS  = int(os.environ.get("SIMMTM_ELAYERS", "3"))

_ModelClass = None   # cache da classe Model do SimMTM (import isolado)


class _Cfg:
    """Config mínima (dot-access) exigida por models.SimMTM.Model.__init__."""
    def __init__(self, seq_len, pred_len):
        self.task_name = "finetune"
        self.seq_len = seq_len
        self.label_len = 0
        self.pred_len = pred_len
        self.enc_in = 1
        self.d_model = _D_MODEL
        self.n_heads = _N_HEADS
        self.d_ff = _D_FF
        self.e_layers = _E_LAYERS
        self.dropout = 0.1
        self.head_dropout = 0.1
        self.factor = 1
        self.activation = "gelu"
        self.embed = "timeF"
        self.freq = "h"
        self.output_attention = False


def _get_model_class():
    global _ModelClass
    if _ModelClass is None:
        _ModelClass = import_model_class(SIMMTM_ROOT, "models.SimMTM", "Model")
    return _ModelClass


def _maybe_load_pretrained(model, device):
    """
    PT: Carrega pesos pré-treinados do SimMTM (chave 'model_state_dict'), copiando
        apenas camadas de nome+shape iguais e IGNORANDO a cabeça de forecast (que é
        re-treinada). Se o checkpoint não existir, segue com fine-tune do zero.
    EN: Loads SimMTM pretrained weights ('model_state_dict'), copying only layers
        with matching name+shape and SKIPPING the forecast head (re-trained). If the
        checkpoint is missing, proceeds with finetune-from-scratch.
    """
    if not os.path.isfile(SIMMTM_CKPT):
        print(f"[SimMTM][WARN] pretrained checkpoint not found at {SIMMTM_CKPT}; "
              f"finetuning from scratch.", flush=True)
        return
    ckpt = torch.load(SIMMTM_CKPT, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    own = model.state_dict()
    matched = 0
    for name, param in own.items():
        if "head" in name:            # cabeça de forecast é nova
            continue
        src = state.get(name, None)
        if src is not None and tuple(src.shape) == tuple(param.shape):
            param.copy_(src)
            matched += 1
    print(f"[SimMTM] transferred {matched} pretrained tensors from {SIMMTM_CKPT}",
          flush=True)


def _make_windows(train_cpu, seq_len, horizon):
    """Gera pares (X:[N,seq_len], Y:[N,horizon]) deslizantes de todas as séries."""
    X, Y = [], []
    for series in train_cpu:
        n = len(series)
        last = n - seq_len - horizon
        if last < 0:
            continue
        for t in range(last + 1):
            X.append(series[t:t + seq_len])
            Y.append(series[t + seq_len:t + seq_len + horizon])
    if not X:
        return None, None
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def predict_simmtm(train_cpu, horizon, context_length, device="cpu", seed=42):
    """
    PT: Ajusta o SimMTM (pré-treino + fine-tune) e prevê `horizon` passos por série.
        - train_cpu: np.ndarray (n_series, context_length), série CRUA.
        - retorna: torch.Tensor (n_series, horizon) no device, espaço CRU.
    EN: Finetunes SimMTM and forecasts `horizon` steps per series.
        - train_cpu: np.ndarray (n_series, context_length), RAW series.
        - returns: torch.Tensor (n_series, horizon) on device, RAW space.
    """
    train_cpu = np.asarray(train_cpu, dtype=np.float32)
    n_series = train_cpu.shape[0]

    # L não pode consumir todo o contexto: precisa sobrar espaço para o alvo.
    seq_len = min(_MAX_SEQ_LEN, context_length - horizon - 1)
    seq_len = max(seq_len, 1)

    torch.manual_seed(seed)
    np.random.seed(seed)

    ModelClass = _get_model_class()
    cfg = _Cfg(seq_len=seq_len, pred_len=horizon)
    model = ModelClass(cfg).to(device)
    _maybe_load_pretrained(model, device)

    X, Y = _make_windows(train_cpu, seq_len, horizon)
    if X is None:
        # Sem janelas suficientes: devolve zeros (fallback tratado no chamador também).
        return torch.zeros((n_series, horizon), device=device)

    X_t = torch.from_numpy(X).unsqueeze(-1).to(device)   # (N, seq_len, 1)
    Y_t = torch.from_numpy(Y).to(device)                 # (N, horizon)

    opt = torch.optim.Adam(model.parameters(), lr=_LR)
    loss_fn = nn.MSELoss()

    model.train()
    n = X_t.shape[0]
    for _ in range(_EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, _BATCH_SIZE):
            idx = perm[i:i + _BATCH_SIZE]
            xb, yb = X_t[idx], Y_t[idx]
            out = model(xb, None)              # (bs, horizon, 1)
            loss = loss_fn(out[:, :, 0], yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Predição: última janela de cada série -> horizon passos à frente.
    model.eval()
    last_windows = torch.from_numpy(
        train_cpu[:, -seq_len:]
    ).unsqueeze(-1).to(device)                 # (n_series, seq_len, 1)

    preds = []
    with torch.no_grad():
        for i in range(0, n_series, _BATCH_SIZE):
            out = model(last_windows[i:i + _BATCH_SIZE], None)   # (bs, horizon, 1)
            preds.append(out[:, :, 0])
    return torch.cat(preds, dim=0)             # (n_series, horizon)
