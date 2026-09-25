"""Shared config / paths for the copula (2-pt with PDF removed) extension."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
sys.path.insert(0, STUDY)

with open(os.path.join(HERE, "config.json")) as f:
    CFG = json.load(f)

SCRATCH = os.environ.get("COPULA_SCRATCH", os.path.join(STUDY, "_scratch", "copula"))
OUT = os.environ.get("COPULA_OUT", os.path.join(HERE, "results"))
FIG = os.environ.get("COPULA_FIG", os.path.join(HERE, "figures"))
for d in (SCRATCH, OUT, FIG):
    os.makedirs(d, exist_ok=True)


def a_of_z(z):
    return 1.0 / (1.0 + z)


def snap_path(model, n, z, seed=None):
    s = "" if seed in (None, CFG["phase_seed"]) else f"_seed{seed}"
    return os.path.join(SCRATCH, f"psi_{model}_N{n}_z{z:g}{s}.npy")
