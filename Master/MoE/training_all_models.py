import os

# ======================================
# CONCURRENCY / STABILITY CONFIG (1 GPU, 16 CPU cores)
# --------------------------------------
# Regra de ouro: N_JOBS * THREADS_PER_WORKER <= núcleos físicos (16).
# Mantém a máquina responsiva e evita explosão de threads / OOM na GPU única.
# ======================================
GPU_ID             = "2"   # índice da única GPU; ajuste se a sua não for a 0
N_TRAIN_JOBS       = 3     # subprocessos de main.py em paralelo (treino)  -> 3*5=15
N_EVAL_JOBS        = 2     # jobs de avaliação em paralelo (run_experiments) -> 2*5=10
THREADS_PER_WORKER = 5     # threads BLAS/OMP por worker
MAX_TRAIN_ATTEMPTS = 3     # nº de tentativas de treino (re-roda os .pt que faltaram)

# ======================================
# GENERAL CONFIGURATION
# ======================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Teto de threads DEVE ser definido ANTES de importar torch/numpy
# (que entram via joblib / run_experiments), senão os pools BLAS já sobem
# do tamanho de todos os núcleos e o limite não é respeitado.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(THREADS_PER_WORKER)

import subprocess
import re
from itertools import product
from joblib import Parallel, delayed
from run_experiments import run_full_experiment_pipeline

base_path = "../all_datasets_global_by_years"
HORIZONS = [3, 6, 12, 24]

BASE_CONTEXT = 410
MAX_YEAR = 2024
MIN_YEAR = 2020

debug_state = None  # None
device = "cuda"

# ======================================
# HYPERPARAMETER GRID
# ======================================
TOP_K_LIST = [2]
NORM_LIST = ["std", "minmax"]
USE_NOISE_LIST = [True]
EPOCHS_LIST = [30, 60, 100]
LR_LIST = [1e-4, 1e-3, 1e-5]

# ======================================
# HELPERS
# ======================================
def get_context_length(year: int, horizon: int) -> int:
    return BASE_CONTEXT - horizon - ((MAX_YEAR - year) * horizon)


def build_experiment_name(top_k, norm, use_noise, epochs, lr):
    return (
        f"topk_{top_k}"
        f"_norm_{norm}"
        f"_noise_{use_noise}"
        f"_ep_{epochs}"
        f"_lr_{lr}"
    )


def is_model_complete(save_model_path: str) -> bool:
    """
    Retorna True se o arquivo .pt específico deste job já existe.
    Isso garante que apenas este job individual seja pulado,
    sem afetar os demais jobs do mesmo experimento.
    """
    return os.path.isfile(save_model_path)


def run_training(
    command,
    excluded_state,
    year,
    horizon,
    context_length,
    top_k,
    norm,
    dataset_path,
    device,
):
    print(
        f"Training: excluding={excluded_state} | "
        f"year={year} | horizon={horizon} | "
        f"context={context_length} | "
        f"top_k={top_k} | "
        f"norm={norm} | "
        f"data={dataset_path} | "
        f"device={device}",
        flush=True
    )

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ},  # herda CUDA_VISIBLE_DEVICES e tetos de thread
        )

        print(
            f"✔ DONE | excluding={excluded_state} | year={year} | horizon={horizon}",
            flush=True
        )

        if result.stdout:
            print(result.stdout, flush=True)

        if result.stderr:
            print(result.stderr, flush=True)

    except subprocess.CalledProcessError as e:
        print(
            f"✖ FAILED | excluding={excluded_state} | year={year} | horizon={horizon}",
            flush=True
        )
        print(e.stderr, flush=True)


# ======================================
# HYPERPARAMETER COMBINATIONS
# ======================================
EXPERIMENTS = list(product(
    TOP_K_LIST,
    NORM_LIST,
    USE_NOISE_LIST,
    EPOCHS_LIST,
    LR_LIST
))


# ======================================
# BUILD JOB LIST (GRID SEARCH)
# ======================================
def build_jobs():
    """
    Varre o grid e retorna:
      - jobs: a lista de jobs de treino que AINDA faltam (pula os .pt já existentes)
      - all_expected_models: todos os .pt que deveriam existir ao final

    Como relê o disco a cada chamada, ao ser chamada novamente após uma rodada
    de treino só retornam em `jobs` os modelos que CONTINUAM faltando — é isso
    que permite o loop de retry re-executar apenas os que falharam.
    """
    jobs = []
    all_expected_models = []

    for top_k, norm, use_noise, epochs, lr in EXPERIMENTS:

        experiment_name = build_experiment_name(
            top_k, norm, use_noise, epochs, lr
        )

        trained_models_root = os.path.join("trained_models", experiment_name)
        os.makedirs(trained_models_root, exist_ok=True)

        for HORIZON in HORIZONS:
            horizon_base_path = os.path.join(base_path, f"horizon_{HORIZON}")
            if not os.path.exists(horizon_base_path):
                continue

            for excluded_state_folder in os.listdir(horizon_base_path):
                if not excluded_state_folder.startswith("excluding_"):
                    continue

                excluded_state = excluded_state_folder.replace("excluding_", "")

                if debug_state is not None and excluded_state != debug_state:
                    continue

                folder_path = os.path.join(horizon_base_path, excluded_state_folder)
                if not os.path.isdir(folder_path):
                    continue

                horizon_dir = os.path.join(trained_models_root, f"horizon_{HORIZON}")
                state_model_dir = os.path.join(horizon_dir, excluded_state_folder)
                os.makedirs(state_model_dir, exist_ok=True)

                for dataset_file in os.listdir(folder_path):
                    if not dataset_file.endswith(".jsonl"):
                        continue

                    match = re.search(r"dataset_(\d{4})\.jsonl", dataset_file)
                    if not match:
                        continue

                    year = int(match.group(1))
                    if year < MIN_YEAR or year > MAX_YEAR:
                        continue

                    context_length = get_context_length(year, HORIZON)
                    if context_length <= 0:
                        continue

                    dataset_path = os.path.join(folder_path, dataset_file)

                    save_model_path = os.path.join(
                        state_model_dir,
                        f"model_{experiment_name}_{year}.pt"
                    )
                    all_expected_models.append(save_model_path)

                    # Pula apenas este job individual se o .pt já existe
                    if is_model_complete(save_model_path):
                        continue

                    command = [
                        "python", "main.py",
                        "--mode", "train",
                        "--data", dataset_path,
                        "--context_length", str(context_length),
                        "--horizon", str(HORIZON),
                        "--top_k", str(top_k),
                        "--use_noise", "true" if use_noise else "false",
                        "--norm", norm,
                        "--epochs", str(epochs),
                        "--lr", str(lr),
                        "--save_path", save_model_path,
                        "--device", device
                    ]

                    jobs.append((
                        command,
                        excluded_state,
                        year,
                        HORIZON,
                        context_length,
                        top_k,
                        norm,
                        dataset_path,
                        device
                    ))

    return jobs, all_expected_models

# ======================================
# RUN TRAINING WITH AUTOMATIC RETRY
# --------------------------------------
# Treina e, ao final de cada rodada, VERIFICA o que faltou e re-executa
# APENAS os modelos ausentes — repetindo até todos existirem OU esgotar
# MAX_TRAIN_ATTEMPTS (o limite evita loop infinito em falha determinística).
# ======================================
all_expected_models = []

for attempt in range(1, MAX_TRAIN_ATTEMPTS + 1):
    jobs, all_expected_models = build_jobs()

    if not jobs:
        print("\nNenhum job de treino pendente — todos os modelos já existem.")
        break

    print(f"\n{'='*60}")
    print(f"  Tentativa de treino {attempt}/{MAX_TRAIN_ATTEMPTS}")
    print(f"  Modelos esperados (total)    : {len(all_expected_models)}")
    print(f"  Modelos a treinar (faltando) : {len(jobs)}")
    print(f"{'='*60}\n")

    Parallel(
        n_jobs=N_TRAIN_JOBS,
        backend="loky",
        verbose=10
    )(
        delayed(run_training)(*job) for job in jobs
    )

    missing_after = [p for p in all_expected_models if not os.path.isfile(p)]
    if not missing_after:
        print(f"\n✔ Todos os {len(all_expected_models)} modelos foram gerados.")
        break

    print(
        f"\nApós a tentativa {attempt}/{MAX_TRAIN_ATTEMPTS}, "
        f"ainda faltam {len(missing_after)} modelos."
        + ("  Re-executando os que faltam...\n"
           if attempt < MAX_TRAIN_ATTEMPTS else "\n")
    )

print("All trainings completed.")

# ======================================
# FINAL RECONCILIATION REPORT
# ======================================
missing_models = [p for p in all_expected_models if not os.path.isfile(p)]
print(f"\n{'='*60}")
print(f"  Modelos esperados (total)   : {len(all_expected_models)}")
print(f"  Modelos presentes           : {len(all_expected_models) - len(missing_models)}")
print(f"  Modelos faltando            : {len(missing_models)}")
print(f"{'='*60}")
if missing_models:
    print(f"  ATENÇÃO: os modelos abaixo NÃO foram gerados mesmo após "
          f"{MAX_TRAIN_ATTEMPTS} tentativas (verifique os logs de erro acima):")
    for p in missing_models:
        print(f"    FALTANDO: {p}")
    print(f"{'='*60}\n")
else:
    print("  Todos os modelos de treino foram gerados com sucesso.")
    print(f"{'='*60}\n")

# ======================================
# RUN ANALYSIS PER EXPERIMENT (sequencial)
# --------------------------------------
# Cada run_full_experiment_pipeline já paraleliza internamente (n_cores).
# Rodar os experimentos em SEQUÊNCIA evita o paralelismo aninhado
# (antes: 3 x 5 = 15 processos pesados na mesma GPU -> travava a máquina).
# ======================================
def run_experiment(top_k, norm, use_noise, epochs, lr):
    print(f"[INICIANDO] top_k={top_k} | norm={norm} | use_noise={use_noise} | epochs={epochs} | lr={lr}")

    experiment_name = build_experiment_name(
        top_k, norm, use_noise, epochs, lr
    )

    trained_models_root = os.path.join(
        "trained_models",
        experiment_name
    )

    run_full_experiment_pipeline(
        path_trained_models=trained_models_root,
        experiment_name="model_" + experiment_name,
        top_k=top_k,
        use_noise=use_noise,
        n_cores=N_EVAL_JOBS,
        norm=norm,
    )

for top_k, norm, use_noise, epochs, lr in EXPERIMENTS:
    run_experiment(top_k, norm, use_noise, epochs, lr)