# =============================================================================
# PT: Utilitário para importar classes de modelo a partir dos repositórios
#     vendorizados (third_party/UniTS e third_party/SimMTM). O problema central é
#     que AMBOS os repos definem pacotes de topo com o MESMO nome (`models`,
#     `layers`, `utils`, `exp`, `data_provider`). Se colocarmos os dois em
#     sys.path simultaneamente, o import de um "sequestra" o do outro.
#
#     `import_model_class` faz o import de forma ISOLADA: salva sys.path/sys.modules,
#     remove os módulos de topo conflitantes, insere a raiz do repo no início do
#     path, importa a classe desejada e então restaura o estado anterior. A classe
#     retornada mantém suas dependências já vinculadas (bound), então continua
#     funcionando depois da restauração.
#
# EN: Helper to import model classes from the vendored repos (third_party/UniTS and
#     third_party/SimMTM). Both repos define top-level packages with the SAME names
#     (`models`, `layers`, `utils`, `exp`, `data_provider`); having both on sys.path
#     at once makes one shadow the other. `import_model_class` imports in ISOLATION:
#     it snapshots sys.path/sys.modules, drops the conflicting top-level modules,
#     prepends the repo root, imports the target class, then restores the previous
#     state. The returned class keeps its dependencies bound, so it works afterwards.
# =============================================================================
import importlib
import os
import sys
import threading

# Raiz do pacote MoE (…/Master/MoE), a partir deste arquivo em setup/baselines/
_MOE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

UNITS_ROOT = os.path.join(_MOE_ROOT, "third_party", "UniTS")
SIMMTM_ROOT = os.path.join(_MOE_ROOT, "third_party", "SimMTM", "SimMTM_Forecasting")

UNITS_CKPT = os.path.join(
    UNITS_ROOT, "checkpoints", "units_x128_pretrain_checkpoint.pth"
)
# SimMTM: pesos pré-treinados vêm do Tsinghua Cloud (download manual). Caminho
# padrão + override por env SIMMTM_CKPT. Se ausente, o wrapper faz fine-tune do zero.
SIMMTM_CKPT = os.environ.get(
    "SIMMTM_CKPT",
    os.path.join(_MOE_ROOT, "third_party", "SimMTM", "checkpoints", "ckpt_best.pth"),
)

# Nomes de pacotes de topo que os dois repos definem e que podem colidir entre si
# (e, por precaução, com o ambiente hospedeiro). São removidos/rest­aurados no import.
_CONFLICTING_TOP_LEVEL = {
    "models", "layers", "utils", "exp", "data_provider", "ns_layers", "data",
}

_import_lock = threading.Lock()


def import_model_class(repo_root: str, dotted_module: str, class_name: str = "Model"):
    """
    PT: Importa `class_name` de `dotted_module` (ex.: 'models.SimMTM') a partir de
        `repo_root`, isolando o import dos pacotes de topo conflitantes.
    EN: Imports `class_name` from `dotted_module` (e.g. 'models.SimMTM') located at
        `repo_root`, isolating the import from conflicting top-level packages.
    """
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"Vendored repo not found: {repo_root}")

    with _import_lock:
        saved_path = list(sys.path)
        # Salva e remove da cache os módulos de topo conflitantes já carregados.
        removed = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k.split(".")[0] in _CONFLICTING_TOP_LEVEL
        }
        sys.path.insert(0, repo_root)
        try:
            module = importlib.import_module(dotted_module)
            cls = getattr(module, class_name)
        finally:
            # Restaura sys.path e limpa o que ESTE import trouxe (evita vazar/colidir),
            # devolvendo os módulos originais do hospedeiro.
            sys.path[:] = saved_path
            for k in [
                k for k in list(sys.modules)
                if k.split(".")[0] in _CONFLICTING_TOP_LEVEL
            ]:
                del sys.modules[k]
            sys.modules.update(removed)
        return cls
