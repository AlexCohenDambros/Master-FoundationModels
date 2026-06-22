"""
Junta as pastas de benchmark "normais" e as "*_old" em uma unica pasta "*_final",
fazendo UNIAO DE ARQUIVOS (preservando o caminho relativo de cada arquivo) e depois
valida, por hash de conteudo, que todo arquivo de origem foi corretamente copiado.

Pares processados (dentro de RESULTADOS_COMPLETOS_BENCHMARK):
    results_benchmark         + results_benchmark_old         -> results_benchmark_final
    times_benchmark           + times_benchmark_old           -> times_benchmark_final
    trained_models_benchmark  + trained_models_benchmark_old  -> trained_models_benchmark_final

Regras de uniao (por arquivo, identificado pelo caminho relativo dentro do par):
    - Arquivo so existe no "novo"  -> copiado como esta.
    - Arquivo so existe no "_old"  -> copiado como esta.
    - Existe nos dois, conteudo IDENTICO -> copiado uma vez so.
    - Existe nos dois, conteudo DIFERENTE -> resolvido conforme --policy:
        keep_both  (padrao): mantem o do "novo" com o nome original e grava o do
                             "_old" ao lado com sufixo "_old" antes da extensao
                             (ex: train_loss.csv -> train_loss_old.csv). Nada e perdido.
        prefer_new          : mantem o do "novo"; o do "_old" e descartado.
        prefer_old          : mantem o do "_old"; o do "novo" e descartado.

Nenhuma pasta original e apagada. As "*_final" sao sempre recriadas do zero.

Uso:
    python merge_benchmark_results.py                 # executa com policy=keep_both
    python merge_benchmark_results.py --dry-run       # apenas mostra o que faria
    python merge_benchmark_results.py --policy prefer_new
    python merge_benchmark_results.py --base "caminho/para/RESULTADOS_COMPLETOS_BENCHMARK"
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

# Cada item: (pasta_nova, pasta_old, pasta_final)
PAIRS = [
    ("results_benchmark", "results_benchmark_old", "results_benchmark_final"),
    ("times_benchmark", "times_benchmark_old", "times_benchmark_final"),
    ("trained_models_benchmark", "trained_models_benchmark_old", "trained_models_benchmark_final"),
]

DEFAULT_BASE = Path(__file__).resolve().parent / "RESULTADOS_COMPLETOS_BENCHMARK"


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 do conteudo do arquivo."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def list_files(root: Path) -> dict[str, Path]:
    """Mapeia caminho_relativo (str com '/') -> Path absoluto, para todos os arquivos."""
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            out[rel] = p
    return out


def with_old_suffix(rel: str) -> str:
    """train_loss.csv -> train_loss_old.csv (sufixo antes da extensao)."""
    p = Path(rel)
    return p.with_name(p.stem + "_old" + p.suffix).as_posix()


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def merge_pair(base: Path, new_name: str, old_name: str, final_name: str,
               policy: str, dry_run: bool) -> dict:
    new_dir = base / new_name
    old_dir = base / old_name
    final_dir = base / final_name

    report = {
        "pair": final_name,
        "new_exists": new_dir.exists(),
        "old_exists": old_dir.exists(),
        "from_new_only": 0,
        "from_old_only": 0,
        "identical": 0,
        "conflict_kept_both": 0,
        "conflict_new_wins": 0,
        "conflict_old_wins": 0,
        "written": 0,
        "missing_sources": [],
        "conflicts": [],
    }

    if not new_dir.exists() and not old_dir.exists():
        report["error"] = f"Nenhuma das pastas existe: {new_dir} / {old_dir}"
        return report

    new_files = list_files(new_dir)
    old_files = list_files(old_dir)
    report["count_new"] = len(new_files)
    report["count_old"] = len(old_files)

    # Recria a pasta final do zero (sem apagar as originais)
    if final_dir.exists() and not dry_run:
        shutil.rmtree(final_dir)
    if not dry_run:
        final_dir.mkdir(parents=True, exist_ok=True)

    all_rels = sorted(set(new_files) | set(old_files))
    for rel in all_rels:
        in_new = rel in new_files
        in_old = rel in old_files
        dest = final_dir / rel

        if in_new and not in_old:
            copy_file(new_files[rel], dest, dry_run)
            report["from_new_only"] += 1
            report["written"] += 1
        elif in_old and not in_new:
            copy_file(old_files[rel], dest, dry_run)
            report["from_old_only"] += 1
            report["written"] += 1
        else:  # existe nos dois
            hn = file_hash(new_files[rel])
            ho = file_hash(old_files[rel])
            if hn == ho:
                copy_file(new_files[rel], dest, dry_run)
                report["identical"] += 1
                report["written"] += 1
            else:
                report["conflicts"].append(rel)
                if policy == "keep_both":
                    copy_file(new_files[rel], dest, dry_run)
                    old_dest = final_dir / with_old_suffix(rel)
                    copy_file(old_files[rel], old_dest, dry_run)
                    report["conflict_kept_both"] += 1
                    report["written"] += 2
                elif policy == "prefer_new":
                    copy_file(new_files[rel], dest, dry_run)
                    report["conflict_new_wins"] += 1
                    report["written"] += 1
                elif policy == "prefer_old":
                    copy_file(old_files[rel], dest, dry_run)
                    report["conflict_old_wins"] += 1
                    report["written"] += 1
    return report


def validate_pair(base: Path, new_name: str, old_name: str, final_name: str,
                  policy: str) -> dict:
    """Confere que todo arquivo de origem esta no final, comparando por hash."""
    new_dir = base / new_name
    old_dir = base / old_name
    final_dir = base / final_name

    result = {
        "pair": final_name,
        "ok": True,
        "checked_new": 0,
        "checked_old": 0,
        "problems": [],
    }

    new_files = list_files(new_dir)
    old_files = list_files(old_dir)
    final_files = list_files(final_dir)
    final_hashes = {rel: file_hash(p) for rel, p in final_files.items()}

    # 1) Todo arquivo do "novo" deve existir igual no final (mesmo caminho).
    for rel, src in new_files.items():
        result["checked_new"] += 1
        if rel not in final_hashes:
            result["ok"] = False
            result["problems"].append(f"[novo] ausente no final: {rel}")
        elif final_hashes[rel] != file_hash(src):
            result["ok"] = False
            result["problems"].append(f"[novo] hash diferente no final: {rel}")

    # 2) Todo arquivo do "_old" deve existir igual no final, considerando a policy.
    for rel, src in old_files.items():
        result["checked_old"] += 1
        ho = file_hash(src)
        conflict = rel in new_files and file_hash(new_files[rel]) != ho

        if not conflict:
            # sem conflito: o old deve estar no caminho normal (mesmo conteudo)
            target = rel
        else:
            if policy == "keep_both":
                target = with_old_suffix(rel)
            elif policy == "prefer_old":
                target = rel
            else:  # prefer_new -> conteudo do old foi descartado de proposito
                if final_hashes.get(rel) == file_hash(new_files[rel]):
                    continue  # comportamento esperado
                result["ok"] = False
                result["problems"].append(f"[old] esperado descartado mas final divergiu: {rel}")
                continue

        if target not in final_hashes:
            result["ok"] = False
            result["problems"].append(f"[old] ausente no final: {target}")
        elif final_hashes[target] != ho:
            result["ok"] = False
            result["problems"].append(f"[old] hash diferente no final: {target}")

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Junta pastas *_benchmark e *_old em *_final e valida.")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE,
                    help="Pasta RESULTADOS_COMPLETOS_BENCHMARK")
    ap.add_argument("--policy", choices=["keep_both", "prefer_new", "prefer_old"],
                    default="keep_both", help="Resolucao de conflito (mesmo arquivo, conteudo diferente)")
    ap.add_argument("--dry-run", action="store_true", help="Nao escreve nada, so mostra o resumo")
    args = ap.parse_args()

    base: Path = args.base.resolve()
    if not base.exists():
        print(f"ERRO: pasta base nao encontrada: {base}")
        return 2

    print(f"Base: {base}")
    print(f"Policy de conflito: {args.policy}")
    print(f"Modo: {'DRY-RUN (nada sera escrito)' if args.dry_run else 'EXECUCAO'}")
    print("=" * 70)

    merge_reports = []
    for new_name, old_name, final_name in PAIRS:
        print(f"\n>>> MERGE: {new_name} + {old_name} -> {final_name}")
        rep = merge_pair(base, new_name, old_name, final_name, args.policy, args.dry_run)
        merge_reports.append(rep)
        if "error" in rep:
            print(f"    ERRO: {rep['error']}")
            continue
        print(f"    arquivos no novo : {rep.get('count_new', 0)}")
        print(f"    arquivos no _old : {rep.get('count_old', 0)}")
        print(f"    so do novo       : {rep['from_new_only']}")
        print(f"    so do _old       : {rep['from_old_only']}")
        print(f"    identicos        : {rep['identical']}")
        if rep["conflicts"]:
            print(f"    CONFLITOS (mesmo caminho, conteudo diferente): {len(rep['conflicts'])}")
            print(f"      -> keep_both: {rep['conflict_kept_both']} | "
                  f"new_wins: {rep['conflict_new_wins']} | old_wins: {rep['conflict_old_wins']}")
            for c in rep["conflicts"][:5]:
                print(f"      - {c}")
            if len(rep["conflicts"]) > 5:
                print(f"      ... (+{len(rep['conflicts']) - 5} outros)")
        print(f"    total escrito no final: {rep['written']}")

    if args.dry_run:
        print("\nDRY-RUN concluido. Rode sem --dry-run para aplicar e validar.")
        return 0

    print("\n" + "=" * 70)
    print("VALIDACAO (comparando final com as origens, por hash de conteudo)")
    print("=" * 70)
    all_ok = True
    for new_name, old_name, final_name in PAIRS:
        res = validate_pair(base, new_name, old_name, final_name, args.policy)
        status = "OK" if res["ok"] else "FALHOU"
        print(f"\n>>> {final_name}: {status}")
        print(f"    conferidos do novo: {res['checked_new']} | do _old: {res['checked_old']}")
        if not res["ok"]:
            all_ok = False
            for prob in res["problems"][:20]:
                print(f"    - {prob}")
            if len(res["problems"]) > 20:
                print(f"    ... (+{len(res['problems']) - 20} outros problemas)")

    print("\n" + "=" * 70)
    if all_ok:
        print("RESULTADO FINAL: TODAS as pastas _final foram validadas com sucesso. "
              "Todo arquivo de origem (novo e _old) esta presente e integro.")
        return 0
    print("RESULTADO FINAL: HOUVE PROBLEMAS na validacao (veja acima).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
