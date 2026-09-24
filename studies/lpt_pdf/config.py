"""Shared study configuration."""
import os

L = 300.0            # box, Mpc/h
R_F = 14.0           # real-space top-hat filter radius applied to the ICs, Mpc/h
SEED = 1             # master phase seed
N_MASTER = 512       # master phase grid (all resolutions are sub-cubes of it)
N_ANA = 256          # common analysis mesh for all density estimates
R_SMOOTH = [7.0, 14.0, 28.0, 42.0]   # top-hat smoothing radii (0.5, 1, 2, 3 R_F), Mpc/h
A_FINAL = 1.0

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SCRATCH = os.environ.get("LPTPDF_SCRATCH", os.path.join(HERE, "_scratch"))
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)
PHASES = os.path.join(SCRATCH, f"phases_{N_MASTER}_seed{SEED}.npy")
