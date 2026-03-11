import os
from collections import defaultdict

BASE_DIR = "trained_models"
HORIZONS = ["horizon_3", "horizon_6", "horizon_12", "horizon_24"]
REQUIRED_FILES = 5  # .pt files needed to consider complete


def count_pt_files(path):
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if f.endswith(".pt"))


def is_complete(path):
    return count_pt_files(path) >= REQUIRED_FILES


def print_bar(pct, width=30):
    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:.1f}%"


def extract_topk(experiment_name: str) -> str:
    """Extrai o valor de top_k do nome do experimento (ex: topk_1_norm_...)."""
    for part in experiment_name.split("_"):
        pass
    parts = experiment_name.split("_")
    for i, part in enumerate(parts):
        if part == "topk" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def main():
    if not os.path.isdir(BASE_DIR):
        print(f"❌ Pasta '{BASE_DIR}' não encontrada. Execute o script na raiz do projeto.")
        return

    experiments = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    if not experiments:
        print(f"Nenhum experimento encontrado em '{BASE_DIR}'.")
        return

    # ── coleta todos os estados presentes em qualquer experimento
    all_states = set()
    for exp in experiments:
        for h in HORIZONS:
            h_path = os.path.join(BASE_DIR, exp, h)
            if os.path.isdir(h_path):
                for s in os.listdir(h_path):
                    if os.path.isdir(os.path.join(h_path, s)):
                        all_states.add(s)
    all_states = sorted(all_states)

    # ── coleta todos os valores de top_k presentes
    all_topk = sorted(set(extract_topk(e) for e in experiments))

    # ── contagens
    total_global = 0
    done_global  = 0

    horizon_total = defaultdict(int)
    horizon_done  = defaultdict(int)

    state_total = defaultdict(int)
    state_done  = defaultdict(int)

    topk_total = defaultdict(int)
    topk_done  = defaultdict(int)

    detail = []  # (exp, horizon, state, done, n_files)

    for exp in experiments:
        topk = extract_topk(exp)
        for h in HORIZONS:
            for s in all_states:
                path = os.path.join(BASE_DIR, exp, h, s)
                if not os.path.isdir(path):
                    continue
                n = count_pt_files(path)
                done = n >= REQUIRED_FILES

                total_global    += 1
                horizon_total[h] += 1
                state_total[s]   += 1
                topk_total[topk] += 1

                if done:
                    done_global      += 1
                    horizon_done[h]  += 1
                    state_done[s]    += 1
                    topk_done[topk]  += 1

                detail.append((exp, h, s, done, n))

    pct = lambda d, t: (d / t * 100) if t else 0.0

    SEP = "═" * 70

    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RELATÓRIO DE EXPERIMENTOS")
    print(SEP)

    # GLOBAL
    p = pct(done_global, total_global)
    print(f"\n{'GLOBAL':}")
    print(f"  Finalizados : {done_global:>5} / {total_global:<5}  {print_bar(p)}")

    # POR TOP_K
    print(f"\n{'POR TOP_K':}")
    for tk in all_topk:
        t = topk_total[tk]
        d = topk_done[tk]
        p = pct(d, t)
        print(f"  top_k={tk:<8}  {d:>5} / {t:<5}  {print_bar(p)}")

    # POR HORIZONTE
    print(f"\n{'POR HORIZONTE':}")
    for h in HORIZONS:
        t = horizon_total[h]
        d = horizon_done[h]
        p = pct(d, t)
        label = h.replace("horizon_", "h=")
        print(f"  {label:<10}  {d:>5} / {t:<5}  {print_bar(p)}")

    # POR ESTADO
    print(f"\n{'POR ESTADO':}")
    for s in all_states:
        t = state_total[s]
        d = state_done[s]
        p = pct(d, t)
        print(f"  {s:<25}  {d:>5} / {t:<5}  {print_bar(p)}")

    # INCOMPLETOS (opcional – mostra quais faltam arquivos)
    incomplete = [(e, h, s, n) for e, h, s, done, n in detail if not done]
    not_started = [(e, h, s, n) for e, h, s, n in incomplete if n == 0]
    partial     = [(e, h, s, n) for e, h, s, n in incomplete if n > 0]

    if incomplete:
        print(f"\nINCOMPLETOS — total: {len(incomplete)}  |  "
              f"Não iniciados (0/{REQUIRED_FILES}): {len(not_started)}  |  "
              f"Parciais (1-{REQUIRED_FILES-1}/{REQUIRED_FILES}): {len(partial)}")

        if partial:
            print(f"\n  ── PARCIAIS ({len(partial)}) ──")
            print(f"  {'Experimento':<55} {'Horizonte':<12} {'Estado':<25} {'Arquivos .pt':>12}")
            print(f"  {'-'*55} {'-'*12} {'-'*25} {'-'*12}")
            for e, h, s, n in partial:
                print(f"  {e:<55} {h:<12} {s:<25} {n:>5} / {REQUIRED_FILES}")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()