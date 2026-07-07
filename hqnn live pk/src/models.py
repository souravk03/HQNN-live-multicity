"""
Model definitions ported to match the POC (meteogram_poc_2.py), noiseless.

Six architectures — LSTM, GRU, ANN (classical) and QLSTM, QGRU, HQNN (hybrid) —
with MODE-SPECIFIC sizes exactly matching the POC's Config table:

                    MV                              UNI
  LSTM  hidden      32                              24
  GRU   hidden      24                              16
  ANN   hidden      [128,64,32]                     [64,32,16]
  QLSTM hidden 32, 6 qubits, qd 2, c_hidden 16      hidden 24, 4 qubits, qd 2, c_hidden 12
  QGRU  hidden 24, 6 qubits, qd 2, c_hidden 12      hidden 16, 4 qubits, qd 2, c_hidden  8
  HQNN  n_q 6, qd 2, classical [64,32]              n_q 6, qd 2, classical [32,16]

Quantum circuit = AngleEmbedding(X) + StronglyEntanglingLayers, noiseless
(svn_scale=1.0, svn_bias=0.0). VQC runs on default.qubit with diff_method
'backprop' (fast on the simulator). forward() returns the raw scaled prediction;
inverse-scaling happens in the pipeline. No post-hoc corrections.

build_model(name, input_size, hparams=None, mode=None) -> nn.Module
"""
import numpy as np
import torch
import torch.nn as nn

try:
    import pennylane as qml
    _HAS_QML = True
except ImportError:
    _HAS_QML = False

from config import QUANTUM_BACKEND


# ---------------------------------------------------------------------------
# POC per-mode configuration table (noiseless)
# ---------------------------------------------------------------------------
def _fmode(mode):
    return "uv" if mode in ("uv", "univariate") else "mv"

POC_CONFIG = {
    "mv": {
        "lstm":  {"hidden_size": 32},
        "gru":   {"hidden_size": 24},
        "ann":   {"hidden_units": (128, 64, 32)},
        "qlstm": {"hidden_size": 32, "n_qubits": 6, "q_depth": 2, "classical_hidden": 16},
        "qgru":  {"hidden_size": 24, "n_qubits": 6, "q_depth": 2, "classical_hidden": 12},
        "hqnn":  {"n_qubits": 6, "q_depth": 2, "classical_hidden": (64, 32)},
    },
    "uv": {
        "lstm":  {"hidden_size": 24},
        "gru":   {"hidden_size": 16},
        "ann":   {"hidden_units": (64, 32, 16)},
        "qlstm": {"hidden_size": 24, "n_qubits": 4, "q_depth": 2, "classical_hidden": 12},
        "qgru":  {"hidden_size": 16, "n_qubits": 4, "q_depth": 2, "classical_hidden": 8},
        "hqnn":  {"n_qubits": 6, "q_depth": 2, "classical_hidden": (32, 16)},
    },
}


# ---------------------------------------------------------------------------
# Quantum circuit factory (matches POC make_vqc, noiseless)
# ---------------------------------------------------------------------------
def _make_vqc(in_f: int, n_q: int, q_d: int):
    """VQC: Linear(in_f->n_q) -> tanh*pi -> AngleEmbedding(X) ->
    StronglyEntanglingLayers -> <Z> per wire. Noiseless. backprop on simulator."""
    if not _HAS_QML:
        raise ImportError("pennylane is required for quantum models")
    dev = qml.device(QUANTUM_BACKEND, wires=n_q, shots=None)
    w_shape = qml.StronglyEntanglingLayers.shape(n_layers=q_d, n_wires=n_q)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def _circ(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_q), rotation="X")
        qml.StronglyEntanglingLayers(weights, wires=range(n_q))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

    class VQC(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(in_f, n_q)
            self.w = nn.Parameter(0.01 * torch.randn(*w_shape))
            self._pi = float(np.pi)

        def forward(self, x):
            # x: (batch, in_f). backprop handles the batch dim — no per-sample loop.
            encoded = torch.tanh(self.proj(x)) * self._pi
            res = _circ(encoded, self.w)
            if isinstance(res, (list, tuple)):
                res = torch.stack(res, dim=-1)
            return res.float()

    return VQC()


# ---------------------------------------------------------------------------
# Classical models (match POC LSTM_Model / GRU_Model / ANN)
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=32, **_):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=24, **_):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc(self.gru(x)[0][:, -1, :])


class ANNModel(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_units=(128, 64, 32), **_):
        super().__init__()
        layers, prev = [], input_size
        for u in hidden_units:
            layers += [nn.Linear(prev, u), nn.ReLU()]
            prev = u
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x[:, -1, :])


# ---------------------------------------------------------------------------
# Hybrid models (match POC QLSTM / QGRU / HQNN, noiseless)
# ---------------------------------------------------------------------------
class QLSTMModel(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=32,
                 n_qubits=6, q_depth=2, classical_hidden=16, **_):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.vqc = _make_vqc(input_size, n_qubits, q_depth)   # VQC sees full input
        self.fc = nn.Linear(n_qubits, classical_hidden)
        self.out = nn.Linear(hidden_size + classical_hidden, output_size)

    def forward(self, x):
        l_out, _ = self.lstm(x)
        q_out = torch.relu(self.fc(self.vqc(x[:, -1, :])))
        return self.out(torch.cat([l_out[:, -1, :], q_out], dim=1))


class QGRUModel(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=24,
                 n_qubits=6, q_depth=2, classical_hidden=12, **_):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.vqc = _make_vqc(input_size, n_qubits, q_depth)
        self.fc = nn.Linear(n_qubits, classical_hidden)
        self.out = nn.Linear(hidden_size + classical_hidden, output_size)

    def forward(self, x):
        g_out, _ = self.gru(x)
        q_out = torch.relu(self.fc(self.vqc(x[:, -1, :])))
        return self.out(torch.cat([g_out[:, -1, :], q_out], dim=1))


class HQNNModel(nn.Module):
    def __init__(self, input_size, output_size=1, n_qubits=6, q_depth=2,
                 classical_hidden=(64, 32), **_):
        super().__init__()
        cu = tuple(classical_hidden)
        # quantum-encoder branch: in -> cu -> n_qubits, then VQC
        q_enc, prev = [], input_size
        for u in cu:
            q_enc += [nn.Linear(prev, u), nn.GELU()]
            prev = u
        q_enc.append(nn.Linear(prev, n_qubits))
        self.pe = nn.Sequential(*q_enc)
        self.vqc = _make_vqc(n_qubits, n_qubits, q_depth)
        # classical-encoder branch: in -> cu
        c_enc, prev = [], input_size
        for u in cu:
            c_enc += [nn.Linear(prev, u), nn.GELU()]
            prev = u
        self.me = nn.Sequential(*c_enc)
        self.head = nn.Sequential(
            nn.Linear(n_qubits + cu[-1], 16), nn.GELU(),
            nn.Linear(16, output_size))
        self._pi = float(np.pi)

    def forward(self, x):
        last = x[:, -1, :]
        q_out = self.vqc(torch.tanh(self.pe(last)) * self._pi)
        c_out = self.me(last)
        return self.head(torch.cat([q_out, c_out], dim=-1))


_REGISTRY = {
    "lstm":  LSTMModel,
    "gru":   GRUModel,
    "qlstm": QLSTMModel,
    "qgru":  QGRUModel,
    "ann":   ANNModel,
    "hqnn":  HQNNModel,
}


def build_model(name: str, input_size: int, hparams: dict | None = None,
                mode: str | None = None) -> nn.Module:
    """Build a model using the POC's mode-specific config as the base, with any
    tuned hyperparameters overriding on top.

    mode: 'multivariate'/'mv' or 'univariate'/'uv' — selects the POC size table.
          Defaults to multivariate sizes if not given.
    Recognized hparam overrides: hidden -> hidden_size (recurrent),
          dropout (ignored by POC arch — POC models have no dropout),
          q_depth, n_qubits.
    """
    fm = _fmode(mode)
    base = dict(POC_CONFIG[fm].get(name, {}))   # POC defaults for this mode+model
    hp = hparams or {}

    # tuned overrides (only where they apply to this architecture)
    if name in ("lstm", "gru", "qlstm", "qgru") and hp.get("hidden") is not None:
        base["hidden_size"] = int(hp["hidden"])
    if name in ("qlstm", "qgru", "hqnn") and hp.get("q_depth") is not None:
        base["q_depth"] = int(hp["q_depth"])
    if name in ("qlstm", "qgru", "hqnn") and hp.get("n_qubits") is not None:
        base["n_qubits"] = int(hp["n_qubits"])

    model = _REGISTRY[name](input_size=input_size, output_size=1, **base)
    # remember exactly what we were built from, so save_assets can persist it and
    # load_assets can reconstruct an identical architecture (no size mismatches).
    try:
        model._iwf_hp = dict(hparams or {})
        model._iwf_mode = mode
    except Exception:
        model._iwf_hp = {}; model._iwf_mode = mode
    return model
