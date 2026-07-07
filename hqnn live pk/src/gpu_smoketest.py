#!/usr/bin/env python3
"""
A6000 GPU smoke test  —  FULLY SELF-CONTAINED.

It does NOT import your project (no config.py / engine.py / models.py needed).
It builds its own tiny GRU and its own PennyLane quantum layer that mirror what
the real pipeline uses (AngleEmbedding + StronglyEntanglingLayers, torch
interface, backprop). Run it from ANYWHERE:

    python gpu_smoketest.py

Only torch + pennylane need to be installed (they already are in your venv).
Finishes in well under a minute — nothing is trained for real.

It answers: will the pipeline use the A6000, for classical AND quantum models,
and which QUANTUM_BACKEND should we use?  Copy the whole output back.
"""

import sys, time, platform, subprocess

LINE = "=" * 72
results = {}


def _hdr(t):   print("\n" + LINE + "\n" + t + "\n" + LINE)
def _ok(n, d=""):   results[n] = (True, d);  print(f"  [PASS] {n}" + (f"  ·  {d}" if d else ""))
def _fail(n, d=""): results[n] = (False, d); print(f"  [FAIL] {n}" + (f"  ·  {d}" if d else ""))
def _warn(n, d=""): results[n] = (None, d);  print(f"  [WARN] {n}" + (f"  ·  {d}" if d else ""))


def nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or out.stderr.strip()
    except Exception as e:
        return f"nvidia-smi unavailable ({e})"


# ---------------------------------------------------------------------------
# 0. environment
# ---------------------------------------------------------------------------
_hdr("0. ENVIRONMENT")
print(f"  python      : {platform.python_version()}  ({platform.system()} {platform.release()})")
try:
    import torch
    print(f"  torch       : {torch.__version__}   (built for CUDA {torch.version.cuda})")
except Exception as e:
    print(f"  torch import FAILED: {e}\n  Cannot continue."); sys.exit(1)
try:
    import pennylane as qml
    print(f"  pennylane   : {qml.__version__}")
except Exception as e:
    qml = None; print(f"  pennylane import FAILED: {e}")
print(f"  nvidia-smi  : {nvidia_smi()}")


# ---------------------------------------------------------------------------
# 1. CUDA availability
# ---------------------------------------------------------------------------
_hdr("1. CUDA AVAILABILITY")
CUDA = torch.cuda.is_available()
if CUDA:
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    _ok("cuda_available", f"{name}  cc{cap[0]}.{cap[1]}  {total_gb:.0f} GB")
    DEV = "cuda"
else:
    _fail("cuda_available", "torch sees NO GPU (CPU-only build or driver not visible).")
    DEV = "cpu"


# ---------------------------------------------------------------------------
# 2. real CUDA compute
# ---------------------------------------------------------------------------
_hdr("2. CUDA TENSOR COMPUTE")
if CUDA:
    try:
        a = torch.randn(2048, 2048, device="cuda"); b = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10): c = a @ b
        torch.cuda.synchronize()
        assert c.is_cuda and torch.isfinite(c).all()
        _ok("cuda_matmul", f"10x 2048^2 matmul in {(time.time()-t0)*1000:.0f} ms on {c.device}")
    except Exception as e:
        _fail("cuda_matmul", f"{type(e).__name__}: {e}")
else:
    _warn("cuda_matmul", "skipped (no CUDA)")


# ---------------------------------------------------------------------------
# 3. classical GRU train step  (self-contained model, mirrors the real one)
# ---------------------------------------------------------------------------
_hdr("3. CLASSICAL MODEL (GRU) -- forward + backward on GPU")
import torch.nn as nn

class TinyGRU(nn.Module):
    def __init__(self, in_f, hid=24):
        super().__init__()
        self.gru = nn.GRU(in_f, hid, batch_first=True)
        self.head = nn.Linear(hid, 1)
    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1])

try:
    SEQ, FEAT = 14, 77
    m = TinyGRU(FEAT, 24).to(DEV)
    x = torch.randn(16, SEQ, FEAT, device=DEV); y = torch.randn(16, 1, device=DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    if CUDA: torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        opt.zero_grad(); loss = ((m(x) - y) ** 2).mean(); loss.backward(); opt.step()
    if CUDA: torch.cuda.synchronize()
    pdev = str(next(m.parameters()).device)
    (_ok if pdev.startswith(DEV) else _fail)(
        "classical_gru_train", f"5 steps in {(time.time()-t0)*1000:.0f} ms · params on {pdev} · loss={loss.item():.4f}")
except Exception as e:
    import traceback; traceback.print_exc(); _fail("classical_gru_train", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 4. PennyLane backends
# ---------------------------------------------------------------------------
_hdr("4. PENNYLANE BACKENDS")
backend_avail = {}
if qml is not None:
    for be in ["default.qubit", "lightning.qubit", "lightning.gpu"]:
        try:
            qml.device(be, wires=2); backend_avail[be] = True; print(f"  available : {be}")
        except Exception as e:
            backend_avail[be] = False; print(f"  MISSING   : {be}  ({str(e)[:80]})")
    if backend_avail.get("lightning.gpu"):
        _ok("pennylane_backends", "lightning.gpu available -> quantum circuits CAN run on CUDA")
    elif backend_avail.get("default.qubit"):
        _warn("pennylane_backends", "no lightning.gpu — circuit sim may stay on CPU")
    else:
        _fail("pennylane_backends", "no usable quantum backend")
else:
    _fail("pennylane_backends", "pennylane not importable")


# ---------------------------------------------------------------------------
# 5. quantum layer per backend — does the circuit run, and where?
# ---------------------------------------------------------------------------
_hdr("5. QUANTUM CIRCUIT -- per-backend forward+backward + GPU memory probe")

def make_qlayer(backend, n_q=4, q_d=2, diff="backprop"):
    dev = qml.device(backend, wires=n_q, shots=None)
    wsh = qml.StronglyEntanglingLayers.shape(n_layers=q_d, n_wires=n_q)
    @qml.qnode(dev, interface="torch", diff_method=diff)
    def circ(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_q), rotation="X")
        qml.StronglyEntanglingLayers(weights, wires=range(n_q))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
    class QLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(77, n_q)
            self.w = nn.Parameter(0.1 * torch.randn(*wsh))
            self.head = nn.Linear(n_q, 1)
        def forward(self, x):
            z = torch.tanh(self.proj(x[:, -1])) * 3.14159
            outs = []
            for i in range(z.shape[0]):
                r = circ(z[i], self.w)
                outs.append(torch.stack(r) if isinstance(r, (list, tuple)) else r)
            q = torch.stack(outs).float()
            return self.head(q)
    return QLayer()

def run_quantum(backend, diff):
    target = "cuda" if (CUDA and backend in ("lightning.gpu", "default.qubit")) else "cpu"
    qm = make_qlayer(backend, diff=diff).to(target)
    x = torch.randn(8, 14, 77, device=target)
    y = torch.randn(8, 1, device=target)
    opt = torch.optim.Adam(qm.parameters(), lr=1e-3)
    base = 0.0
    if CUDA:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    t0 = time.time()
    for _ in range(3):
        opt.zero_grad(); loss = ((qm(x) - y) ** 2).mean(); loss.backward(); opt.step()
    if CUDA: torch.cuda.synchronize()
    dt = time.time() - t0
    peak_mb = ((torch.cuda.max_memory_allocated() - base) / 1024**2) if CUDA else 0.0
    return dt, peak_mb, loss.item()

if qml is not None:
    if backend_avail.get("default.qubit"):
        try:
            dt, peak, ls = run_quantum("default.qubit", "backprop")
            tag = f"{dt*1000:.0f} ms/3steps · peak GPU {peak:.1f} MB · loss={ls:.3f}"
            if CUDA and peak > 1.0:
                _ok("quantum_default_qubit", f"default.qubit used the GPU · {tag}")
            elif CUDA:
                _warn("quantum_default_qubit", f"default.qubit looks CPU-bound (~0 GPU mem) · {tag}")
            else:
                _warn("quantum_default_qubit", f"ran on CPU (no GPU) · {tag}")
        except Exception as e:
            import traceback; traceback.print_exc(); _fail("quantum_default_qubit", f"{type(e).__name__}: {e}")

    if backend_avail.get("lightning.gpu"):
        try:
            dt, peak, ls = run_quantum("lightning.gpu", "adjoint")
            tag = f"{dt*1000:.0f} ms/3steps · peak GPU {peak:.1f} MB · loss={ls:.3f}"
            if peak > 0.5:
                _ok("quantum_lightning_gpu", f"lightning.gpu ran the circuit on CUDA · {tag}")
            else:
                _warn("quantum_lightning_gpu", f"lightning.gpu ran but GPU mem ~0 · {tag}")
        except Exception as e:
            import traceback; traceback.print_exc()
            _fail("quantum_lightning_gpu", f"{type(e).__name__}: {str(e)[:120]}")
    else:
        _warn("quantum_lightning_gpu", "lightning.gpu not installed")
else:
    _fail("quantum_circuit", "pennylane not importable")


# ---------------------------------------------------------------------------
# 6. snapshot
# ---------------------------------------------------------------------------
_hdr("6. GPU SNAPSHOT (after tests)")
print(f"  nvidia-smi : {nvidia_smi()}")
if CUDA:
    print(f"  torch peak this run: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB "
          f"of {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB")


# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
_hdr("VERDICT")
p = sum(1 for ok, _ in results.values() if ok is True)
w = sum(1 for ok, _ in results.values() if ok is None)
f = sum(1 for ok, _ in results.values() if ok is False)
print(f"  passed: {p}   warned: {w}   failed: {f}\n")

classical_ok = results.get("classical_gru_train", (False,))[0] is True and CUDA
dq = results.get("quantum_default_qubit", (None,))[0]
lg = results.get("quantum_lightning_gpu", (None,))[0]

if not CUDA:
    print("  > No CUDA visible — pipeline would run on CPU. Fix torch/driver, re-run.")
else:
    print("  > CLASSICAL (lstm/gru/ann): "
          + ("will train on the A6000. [OK]" if classical_ok else "NOT GPU-ready — see checks. [X]"))
    if dq is True:
        print("  > QUANTUM with default.qubit: already using the GPU. Keep QUANTUM_BACKEND='default.qubit'. [OK]")
    elif lg is True:
        print("  > QUANTUM: default.qubit is CPU-bound, but lightning.gpu runs the circuit on CUDA. [OK]")
        print("    RECOMMEND: set QUANTUM_BACKEND='lightning.gpu' (diff_method 'adjoint') for the pipeline.")
    elif lg is None and results.get("pennylane_backends", (None,))[0]:
        print("  > QUANTUM: lightning.gpu present but didn't clearly use GPU — see check 5 timings.")
    else:
        print("  > QUANTUM: no GPU path confirmed — quantum models would be CPU-bound (slowest part).")

print("\n  Copy this ENTIRE output back so I can lock the backend and build the pipeline.")
print(LINE)