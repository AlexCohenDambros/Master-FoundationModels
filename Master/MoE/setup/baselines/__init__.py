# =============================================================================
# PT: Pacote de baselines "vendorizados" (repos de pesquisa que NÃO têm pipeline
#     pronto no Hugging Face). Cada módulo expõe uma função `predict_*` que recebe
#     as séries no mesmo formato usado por run_experiments*.py e devolve um tensor
#     (n_series, horizon), para ser registrado junto dos demais modelos.
#
# EN: Package of vendored baselines (research repos without a ready-made Hugging
#     Face pipeline). Each module exposes a `predict_*` function that takes the
#     series in the same format used by run_experiments*.py and returns a tensor
#     of shape (n_series, horizon), to be registered alongside the other models.
# =============================================================================
