# Baselines vendorizados: UniTS e SimMTM

Dois modelos de forecasting que **não** têm pipeline pronto no Hugging Face e por
isso são usados a partir dos repositórios oficiais (vendorizados em
`Master/MoE/third_party/`). Cada wrapper devolve `(n_series, horizon)` no mesmo
formato dos demais modelos de `run_experiments.py` / `run_experiments_benchmark.py`,
e ambos entram na média do `Ensemble`.

| Modelo | Regime | Espaço de saída | Wrapper |
|--------|--------|-----------------|---------|
| **UniTS** (`mims-harvard/UniTS`, NeurIPS'24) | **zero-shot** (checkpoint `units_x128`) | normalizado (chamador faz `*std+mean`) | `units_baseline.predict_units` |
| **SimMTM** (`thuml/SimMTM`, NeurIPS'23) | **fine-tune por caso** (encoder pré-treinado opcional) | cru (normaliza internamente) | `simmtm_baseline.predict_simmtm` |

## Dependências (instalar uma vez)

```bash
pip install timm reformer_pytorch
```
`timm` é exigido pelo UniTS; `reformer_pytorch` é importado pelas camadas do SimMTM.
(Isso subiu `einops` para 0.8.x — o Moirai/uni2ts continua importando normalmente.)

## Checkpoints

- **UniTS** (`units_x128_pretrain_checkpoint.pth`, ~96 MB) — baixado para
  `third_party/UniTS/checkpoints/`. Para re-baixar:
  ```bash
  curl -L -o third_party/UniTS/checkpoints/units_x128_pretrain_checkpoint.pth \
    https://github.com/mims-harvard/UniTS/releases/download/ckpt/units_x128_pretrain_checkpoint.pth
  ```
- **SimMTM** — os pesos pré-treinados ficam no **Tsinghua Cloud** (link no README do
  repo) e exigem **download manual**. Coloque o arquivo em
  `third_party/SimMTM/checkpoints/ckpt_best.pth` (ou aponte a env `SIMMTM_CKPT`).
  **Sem o checkpoint, o SimMTM roda mesmo assim**, fazendo fine-tune do zero (imprime
  um `[SimMTM][WARN]`). A arquitetura padrão (`d_model=32, n_heads=16, d_ff=64,
  e_layers=3`) casa com os checkpoints estilo ETT; ajuste as envs `SIMMTM_*` se usar
  um checkpoint de outra configuração.

> Os pesos (`*.pth`) e a pasta `checkpoints/` estão no `.gitignore`.

## Variáveis de ambiente (opcionais)

| Env | Padrão | Efeito |
|-----|--------|--------|
| `UNITS_SEQ_LEN` | 256 | comprimento de contexto alimentado ao UniTS (múltiplo de 16, ≤ menor contexto) |
| `UNITS_WARM_DATASET` | ETTh1 | tarefa de forecasting usada p/ warm-start dos tokens |
| `SIMMTM_CKPT` | `third_party/SimMTM/checkpoints/ckpt_best.pth` | caminho do checkpoint SimMTM |
| `SIMMTM_SEQ_LEN` | 96 | janela de entrada do fine-tune |
| `SIMMTM_EPOCHS` | 10 | épocas de fine-tune |
| `SIMMTM_BATCH` / `SIMMTM_LR` | 64 / 1e-4 | batch e learning rate |
| `SIMMTM_DMODEL/NHEADS/DFF/ELAYERS` | 32/16/64/3 | arquitetura (casar com o checkpoint) |

## Notas de implementação

- **Import isolado** (`_vendor.import_model_class`): os dois repos definem pacotes de
  topo homônimos (`models`, `layers`, `utils`); o helper isola `sys.path`/`sys.modules`
  no import para evitar colisão.
- **UniTS e horizontes curtos**: o `forecast` do UniTS quebra para `horizon < patch_len`
  (16). O wrapper prevê um comprimento interno múltiplo de 16 e trunca para `horizon`.
- **UniTS zero-shot**: usa `models.UniTS_zeroshot` (tokens compartilhados) + warm-start
  dos tokens a partir de `prompt_tokens.ETTh1`/`mask_tokens.ETTh1` (média sobre canais).
- **SimMTM**: cada série é univariada (channel-independent); o fine-tune usa janelas
  deslizantes de todas as séries do caso (treino conjunto, estilo PatchTST).
- Cada bloco nos scripts tem `try/except` com fallback em zeros, para não derrubar a
  linha inteira do experimento.
