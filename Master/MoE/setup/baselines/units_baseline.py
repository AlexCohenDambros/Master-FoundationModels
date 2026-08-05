# =============================================================================
# PT: Wrapper do UniTS (mims-harvard/UniTS, NeurIPS 2024) para forecasting
#     univariado ZERO-SHOT, usando o checkpoint pré-treinado `units_x128`.
#     Usamos a variante `models.UniTS_zeroshot`, cujos tokens de prompt/mask são
#     COMPARTILHADOS (não keyed por dataset) e cujo `DynamicLinear`/`ForecastHead`
#     interpolam para comprimentos arbitrários — exatamente o caminho de zero-shot
#     "new length" do repo. O backbone/heads pesados são carregados do checkpoint
#     (strict=False) e os tokens compartilhados recebem warm-start a partir de uma
#     tarefa de forecasting pré-treinada (ETTh1, média sobre os canais).
#
#     Detalhe importante: o `forecast` do UniTS quebra para horizontes menores que
#     `patch_len` (=16). Por isso prevemos um comprimento interno múltiplo de 16
#     (>= horizon) e truncamos para `horizon`.
#
#     Entrada é a série JÁ normalizada (como os demais foundation models zero-shot
#     do pipeline); o UniTS re-normaliza/denormaliza internamente, então a saída
#     volta no mesmo espaço da entrada e o CHAMADOR desnormaliza com std/mean.
#
# EN: UniTS wrapper for ZERO-SHOT univariate forecasting using the pretrained
#     `units_x128` checkpoint. We use `models.UniTS_zeroshot`, whose prompt/mask
#     tokens are SHARED (not dataset-keyed) and whose DynamicLinear/ForecastHead
#     interpolate to arbitrary lengths — the repo's "new length" zero-shot path.
#     Heavy backbone/heads load from the checkpoint (strict=False); shared tokens
#     are warm-started from a pretrained forecasting task (ETTh1, mean over channels).
#     UniTS.forecast breaks for horizons smaller than patch_len (=16), so we predict
#     an internal length that is a multiple of 16 (>= horizon) and truncate.
# =============================================================================
import os
import urllib.request
import torch

from ._vendor import import_model_class, UNITS_ROOT, UNITS_CKPT

# URL do release oficial (units_x128). Override por env UNITS_CKPT_URL.
_CKPT_URL = os.environ.get(
    "UNITS_CKPT_URL",
    "https://github.com/mims-harvard/UniTS/releases/download/ckpt/units_x128_pretrain_checkpoint.pth",
)
# O arquivo real tem ~96MB; menor que isso indica ponteiro/redirect/truncado.
_MIN_CKPT_BYTES = 10_000_000

_SEQ_LEN   = int(os.environ.get("UNITS_SEQ_LEN", "256"))   # múltiplo de 16; <= menor contexto
_PATCH_LEN = 16
_STRIDE    = 16
_D_MODEL   = 128
_PROMPT_NUM = 10
_E_LAYERS  = 3
_N_HEADS   = 8
_WARM_DATASET = os.environ.get("UNITS_WARM_DATASET", "ETTh1")  # tarefa p/ warm-start

_ModelClass = None      # classe Model do UniTS_zeroshot (import isolado)
_clean_state = None     # state_dict limpo do checkpoint (cache)
_raw_state = None       # state_dict cru (com tokens keyed) p/ warm-start
_models = {}            # cache de modelos por (seq_len, horizon)


class _Args:
    d_model = _D_MODEL
    n_heads = _N_HEADS
    e_layers = _E_LAYERS
    prompt_num = _PROMPT_NUM
    patch_len = _PATCH_LEN
    stride = _STRIDE
    dropout = 0.1


def _internal_pred_len(horizon):
    # múltiplo de patch_len, com folga suficiente para o cálculo de tokens (>=1).
    return (horizon // _PATCH_LEN + 2) * _PATCH_LEN


def _valid_ckpt(path):
    """True se o arquivo existe, tem tamanho plausível e começa com o magic ZIP (PK)."""
    try:
        if os.path.getsize(path) < _MIN_CKPT_BYTES:
            return False
        with open(path, "rb") as f:
            return f.read(2) == b"PK"
    except OSError:
        return False


def _ensure_checkpoint():
    """
    PT: Garante um checkpoint UniTS VÁLIDO em UNITS_CKPT. Se estiver ausente ou
        corrompido (download truncado, ou página de redirect salva sem `curl -L`,
        que faz o torch.load falhar com "filename 'storages' not found"), tenta
        baixar do release oficial (download atômico via arquivo temporário).
        Sem internet no nó, troca o erro críptico por uma mensagem acionável.
    """
    if _valid_ckpt(UNITS_CKPT):
        return
    os.makedirs(os.path.dirname(UNITS_CKPT), exist_ok=True)
    if os.path.isfile(UNITS_CKPT):
        print(f"[UniTS][WARN] checkpoint invalido/corrompido em {UNITS_CKPT}; "
              f"rebaixando de {_CKPT_URL}", flush=True)
    else:
        print(f"[UniTS] checkpoint ausente; baixando (~96MB) de {_CKPT_URL}", flush=True)
    tmp = f"{UNITS_CKPT}.tmp.{os.getpid()}"
    try:
        urllib.request.urlretrieve(_CKPT_URL, tmp)
        if not _valid_ckpt(tmp):
            raise IOError("arquivo baixado nao e um checkpoint valido (verifique URL/rede).")
        os.replace(tmp, UNITS_CKPT)   # atomico
        print(f"[UniTS] checkpoint salvo em {UNITS_CKPT}", flush=True)
    except Exception as e:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
        # outro worker (loky) pode ter corrigido o arquivo enquanto tentavamos
        if _valid_ckpt(UNITS_CKPT):
            return
        raise FileNotFoundError(
            f"Checkpoint UniTS ausente/corrompido em {UNITS_CKPT} e o download automatico "
            f"falhou ({e}). Baixe manualmente NUM NO COM INTERNET e confirme ~96MB:\n"
            f"  curl -L -o {UNITS_CKPT} {_CKPT_URL}\n"
            f"(sem o -L, o curl salva a pagina de redirect e o torch.load falha com "
            f"\"filename 'storages' not found\")."
        )


def _load_states():
    global _clean_state, _raw_state
    if _clean_state is not None:
        return
    _ensure_checkpoint()
    try:
        ck = torch.load(UNITS_CKPT, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Falha ao ler o checkpoint UniTS em {UNITS_CKPT} ({e}). "
            f"Provavelmente esta corrompido/truncado — rebaixe com: "
            f"curl -L -o {UNITS_CKPT} {_CKPT_URL}"
        )
    sd = ck["student"] if isinstance(ck, dict) and "student" in ck else ck
    # remove prefixo DDP 'module.'
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    _raw_state = sd
    # remove tokens keyed por dataset (o modelo zeroshot usa tokens compartilhados)
    # e quaisquer 'cls_prompts'.
    _clean_state = {
        k: v for k, v in sd.items()
        if not (k.startswith("prompt_tokens.") or k.startswith("mask_tokens.")
                or k.startswith("cls_tokens.") or k.startswith("category_tokens.")
                or "cls_prompts" in k)
    }


def _get_model(seq_len, horizon, device):
    key = (seq_len, horizon)
    if key in _models:
        return _models[key]

    _load_states()
    assert _clean_state is not None and _raw_state is not None
    ModelClass = _get_model_class()

    internal_pred = _internal_pred_len(horizon)
    task_cfg = {
        "task_name": "long_term_forecast",
        "dataset": "CustomUni",
        "data": "custom",
        "features": "M",
        "seq_len": seq_len,
        "pred_len": internal_pred,
        "label_len": 0,
        "enc_in": 1,
    }
    configs_list = [["CustomUni_task", task_cfg]]

    model = ModelClass(_Args(), configs_list, pretrain=False).to(device)
    model.load_state_dict(_clean_state, strict=False)

    # Warm-start dos tokens compartilhados a partir de uma tarefa de forecasting
    # pré-treinada (média sobre a dimensão de canais -> 1 canal).
    pk = f"prompt_tokens.{_WARM_DATASET}"
    mk = f"mask_tokens.{_WARM_DATASET}"
    if pk in _raw_state:
        model.prompt_token.data.copy_(_raw_state[pk].mean(dim=1, keepdim=True).to(device))
    if mk in _raw_state:
        model.mask_token.data.copy_(_raw_state[mk].mean(dim=1, keepdim=True).to(device))

    model.eval()
    _models[key] = model
    return model


def _get_model_class():
    global _ModelClass
    if _ModelClass is None:
        _ModelClass = import_model_class(UNITS_ROOT, "models.UniTS_zeroshot", "Model")
    return _ModelClass


def predict_units(series_scaled, horizon, context_length=None, device="cpu", batch_size=128):
    """
    PT: Forecasting zero-shot com UniTS.
        - series_scaled: torch.Tensor (n_series, L) JÁ normalizado (std) pelo chamador.
        - retorna: torch.Tensor (n_series, horizon) no ESPAÇO normalizado
          (o chamador desnormaliza com std/mean, como faz com os outros zero-shot).
    EN: Zero-shot forecasting with UniTS.
        - series_scaled: torch.Tensor (n_series, L) already std-normalized by caller.
        - returns: torch.Tensor (n_series, horizon) in normalized space.
    """
    x = torch.as_tensor(series_scaled, dtype=torch.float32, device=device)
    if x.dim() == 3:      # (n, L, 1) -> (n, L)
        x = x.squeeze(-1)
    n_series, total_len = x.shape

    seq_len = min(_SEQ_LEN, total_len)
    model = _get_model(seq_len, horizon, device)

    x = x[:, -seq_len:].unsqueeze(-1)          # (n_series, seq_len, 1)

    preds = []
    with torch.no_grad():
        for i in range(0, n_series, batch_size):
            xb = x[i:i + batch_size]
            out = model.forecast(xb, None, 0)  # (bs, internal_pred, 1)
            preds.append(out[:, :horizon, 0])  # trunca para horizon
    return torch.cat(preds, dim=0)             # (n_series, horizon)
