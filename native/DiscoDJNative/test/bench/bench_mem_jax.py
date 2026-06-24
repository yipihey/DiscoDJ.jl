"""Memory benchmark for the JAX DISCO-DJ nLPT field computation.

Reports a CUDA-context baseline (nvidia-smi after a trivial op) and the LPT
working set from JAX's own peak_bytes_in_use (the XLA allocator peak, which
includes cuFFT scratch).  total = base + algo.

  XLA_PYTHON_CLIENT_PREALLOCATE=false python bench_lpt_mem.py <res> <n_order>
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys, subprocess
import jax, jax.numpy as jnp
import discodj

res, order = int(sys.argv[1]), int(sys.argv[2])
L = 1000.0

def gpu_used():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits", "-i", "0"])
    return int(out.split()[0])

try:
    jnp.ones(1).block_until_ready()          # init CUDA context
    base = gpu_used()
    base_dj = (discodj.DiscoDJ(dim=3, res=res, boxsize=L, cosmo="Planck18EEBAOSN",
                               device="gpu", precision="single")
               .with_timetables()
               .with_linear_ps(transfer_function="Eisenstein-Hu")
               .with_ics(seed=42))

    def one():
        dj = base_dj.with_lpt(n_order=order)
        p = dj.evaluate_lpt_psi_at_a(0.02)
        p.block_until_ready()
        return p

    for _ in range(4):
        one()
    algo = int(jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2**20)
    print(f"RESULT res={res} order={order} store=jax base_MiB={base} total_MiB={base + algo}", flush=True)
except Exception as e:
    msg = str(e).lower()
    oom = "out of memory" in msg or "resource_exhausted" in msg or "alloc" in msg
    print(f"RESULT res={res} order={order} store=jax total_MiB={'OOM' if oom else 'ERR'}", flush=True)
