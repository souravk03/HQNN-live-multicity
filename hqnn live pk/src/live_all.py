"""
Whole-country runner for the Map page.

run_all_cycle_stream(...) runs ONE forward cycle for EVERY trained (state, mode)
pair — i.e. both Multivariate and Univariate, for every state that has a trained
model — one after another. Each (state, mode) keeps its OWN cursor (separate
live_state file), so Multivariate Delhi and Univariate Delhi advance
independently: running multi for Jan 1 does NOT push uni to Jan 2.

Events the Map page reacts to:
    {"type":"city_start", "city","label","mode","index","total"}
    {"type":"city_skip",  "city","label","reason"}     # no trained model in any mode
    {"type":"city_done",  "city","label","mode"}
    {"type":"city_error", "city","label","mode","error"}
    {"type":"all_done",   "n","total"}

Inner per-cycle events from live_engine.run_cycle are forwarded too (tagged with
city + mode) so a watching dashboard can animate; the Map only needs the city_*
summary events above.

A (state, mode) counts as trained when its metadata.json exists. Right now Delhi,
Mumbai and Chennai are the live states; this scales to the whole map with no code
changes as more states get trained.
"""
import live_engine
from config import STATES, CITIES, MODES_ENABLED, DEFAULT_MODE


def _label(key):
    cfg = STATES.get(key) or CITIES.get(key) or {}
    return cfg.get("label", key)


def _ordered_states():
    seen, ordered = set(), []
    for k in list(STATES.keys()):
        if k not in seen:
            seen.add(k); ordered.append(k)
    return ordered


def _trained_pairs(modes):
    """List of (state, mode) that have a trained model (metadata.json present)."""
    import paths
    pairs = []
    for k in _ordered_states():
        for mode in modes:
            try:
                if paths.metadata_path(k, mode).exists():
                    pairs.append((k, mode))
            except Exception:
                pass
    return pairs


def run_all_cycle_stream(mode=None, models=None):
    # Always run BOTH enabled modes (multi + uni). The `mode` arg is ignored on
    # purpose: the Map's daily button advances every state in every mode at once.
    modes = list(MODES_ENABLED) or [DEFAULT_MODE]
    pairs = _trained_pairs(modes)
    total = len(pairs)
    trained_states = sorted({s for s, _ in pairs})

    yield {"type": "info",
           "msg": f"Running daily cycle for {len(trained_states)} state(s) x "
                  f"{len(modes)} mode(s) = {total} run(s). Others are skipped."}

    for k in _ordered_states():
        if k not in trained_states:
            yield {"type": "city_skip", "city": k, "label": _label(k),
                   "reason": "no model"}

    done = 0
    for i, (key, m) in enumerate(pairs, start=1):
        label = _label(key)
        mlabel = "multivariate" if m in ("mv", "multivariate") else "univariate"
        yield {"type": "city_start", "city": key, "label": f"{label} - {mlabel}",
               "mode": m, "index": i, "total": total}
        try:
            for ev in live_engine.run_cycle(models=models, mode=m, city=key):
                if isinstance(ev, dict):
                    ev = dict(ev); ev.setdefault("city", key); ev.setdefault("mode", m)
                yield ev
            yield {"type": "city_done", "city": key, "label": f"{label} - {mlabel}", "mode": m}
            done += 1
        except Exception as e:
            yield {"type": "city_error", "city": key, "label": f"{label} - {mlabel}",
                   "mode": m, "error": f"{type(e).__name__}: {e}"}

    yield {"type": "all_done", "n": done, "total": total}
