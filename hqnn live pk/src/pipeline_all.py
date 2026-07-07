#!/usr/bin/env python3
"""
Headless full-build pipeline  —  download -> tune -> train, sequentially, for
every state/UT and both variates. Designed to run unattended on the A6000.

    python src/pipeline_all.py download      # NASA data for all 36 states/UTs
    python src/pipeline_all.py tune          # full Optuna tuning, all states x modes
    python src/pipeline_all.py train         # train all models, all states x modes
    python src/pipeline_all.py all           # download -> tune -> train, in order

    python src/pipeline_all.py cycle 10/05/2026   # run daily cycles up to that date
                                                  # (DD/MM/YYYY or YYYY-MM-DD)

The first four commands do the one-time build. `cycle` advances each TRAINED
state forward (forecast + verify, one day at a time) until its cursor reaches the
given date — resumable from wherever each state currently sits.

Key properties
  • SEQUENTIAL  — one (state x mode) at a time, so no GPU/CPU contention and no
    races on the per-city files. This is the safe way to drive a long build.
  • RESUMABLE   — skips work already done (existing nasa_cache / best_hparams.json
    / model weights). Stop with Ctrl-C and re-run; it picks up where it left off.
    Tuning also resumes mid-study (Optuna storage), training mid-model (checkpoints).
  • ROBUST      — a failure on one item is logged and the run continues.
  • OBSERVABLE  — rolling ETA + projected finish clock, per-item & cumulative
    timing, and a periodic GPU/CPU/RAM line so you can see what the box is doing.
  • Progress is also written to logs/pipeline_progress.json.

Common flags
  --states a,b,c | all     (default: all)
  --modes  multivariate,univariate   (default: both)
  --models lstm,qgru,...   (default: all trainable)
  --trials N               (override Optuna trials for `tune`)
  --force                  (re-do even if already done)
  --device auto|cuda|cpu   (default: auto -> uses the GPU if present)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timedelta

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)


# ───────────────────────── formatting helpers ──────────────────────────
def fmt_dur(sec):
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def clock(dt=None):
    return (dt or datetime.now()).strftime("%H:%M:%S")


def clock_day(dt):
    # include day if the finish is not today
    return dt.strftime("%a %H:%M") if dt.date() != datetime.now().date() else dt.strftime("%H:%M:%S")


def parse_date(s):
    """Accepts DD/MM/YYYY (Indian, tried first), YYYY-MM-DD, DD-MM-YYYY, MM/DD/YYYY.
    Returns a datetime. Raises ValueError with guidance if nothing matches."""
    s = s.strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"could not parse date '{s}' — use DD/MM/YYYY or YYYY-MM-DD")


# ───────────────────────── system monitor ──────────────────────────────
def gpu_line():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        rows = [r for r in out.stdout.strip().splitlines() if r.strip()]
        parts = []
        for i, r in enumerate(rows):
            u, mu, mt, t = [x.strip() for x in r.split(",")]
            parts.append(f"GPU{i} {u}% {mu}/{mt}MB {t}C")
        return " · ".join(parts) if parts else "GPU n/a"
    except Exception:
        return "GPU n/a"


def cpu_line():
    try:
        import psutil
        return f"CPU {psutil.cpu_percent(interval=0.2):.0f}% · RAM {psutil.virtual_memory().percent:.0f}%"
    except Exception:
        try:
            la = os.getloadavg()[0]
            return f"load {la:.1f}"
        except Exception:
            return "cpu n/a"


def sysline():
    return f"   sys: {gpu_line()} | {cpu_line()}"


# ───────────────────────── progress file ───────────────────────────────
class Progress:
    def __init__(self, command):
        self.path = os.path.join(_ROOT, "logs", "pipeline_progress.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.data = {"command": command, "started": datetime.now().isoformat(),
                     "units": []}
        self._flush()

    def record(self, **kw):
        kw["ts"] = datetime.now().isoformat()
        self.data["units"].append(kw)
        self._flush()

    def _flush(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass


# ───────────────────────── skip-if-done checks ─────────────────────────
def download_done(city):
    import paths
    p = paths.nasa_cache(city)
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False


def tune_done(city, mode):
    import paths
    try:
        return paths.best_hparams_path(city, mode).exists()
    except Exception:
        return False


def train_done(city, mode, models, targets):
    import paths
    try:
        for m in models:
            for t in targets:
                if not paths.weights_path(city, mode, m, t).exists():
                    return False
        return True
    except Exception:
        return False


# ───────────────────────── unit runners ────────────────────────────────
def run_download(city):
    import nasa
    df = nasa.download_city(city)
    return f"{len(df)} rows"


def run_tune(city, mode, trials, models):
    import tune
    last = {}
    for ev in tune.tune_stream(city, n_trials=trials, models=models, mode=mode):
        ty = ev.get("type")
        if ty == "tune_model" and ev.get("status") == "done":
            mt = f"{ev.get('model')}/{ev.get('target','')}".rstrip("/")
            v = ev.get("val_mse", ev.get("best_val"))
            print(f"      tuned {mt:18s} best_val_mse={v}", flush=True)
        elif ty == "paused":
            raise RuntimeError("tuning paused/aborted")
        elif ty == "error":
            raise RuntimeError(ev.get("message", "tune error"))
        last = ev
    return "hparams written"


def run_train(city, mode, models):
    import pipeline_stream
    # train_stream prints its own concise "[train] ... done" line per (model,target)
    for ev in pipeline_stream.train_stream(city, models=models, mode=mode):
        ty = ev.get("type")
        if ty == "error":
            raise RuntimeError(ev.get("message", "train error"))
        if ty == "paused":
            raise RuntimeError("training paused/aborted")
    return "weights written"


# ───────────────────────── phase driver ────────────────────────────────
def drive(phase, units, fn, is_done, force, prog):
    """units: list of (label, *args). fn(*args)->detail. is_done(*args)->bool."""
    # decide up-front which units will actually run, for an accurate ETA
    todo = []
    skipped = 0
    for label, args in units:
        if not force and is_done(*args):
            skipped += 1
        else:
            todo.append((label, args))
    total_run = len(todo)
    print(f"\n{'='*72}\nPHASE: {phase}   ·   {len(units)} units total   ·   "
          f"{total_run} to run   ·   {skipped} already done\n{'='*72}")
    if total_run == 0:
        print("  nothing to do — all units already complete.")
        return {"run": 0, "skipped": skipped, "failed": 0, "failures": []}

    run_times = []
    failed = []
    t_phase = time.time()
    for i, (label, args) in enumerate(todo, 1):
        avg = (sum(run_times) / len(run_times)) if run_times else None
        remaining = total_run - (i - 1)
        eta = (avg * remaining) if avg else None
        eta_txt = (f"ETA {fmt_dur(eta)} · finish ~{clock_day(datetime.now()+timedelta(seconds=eta))}"
                   if eta else "ETA —")
        print(f"\n[{i}/{total_run}] {phase}: {label}    ({eta_txt})")
        print(sysline(), flush=True)
        t0 = time.time()
        try:
            detail = fn(*args)
            dt = time.time() - t0
            run_times.append(dt)
            print(f"   ✓ done in {fmt_dur(dt)}   ({detail})", flush=True)
            prog.record(phase=phase, unit=label, status="done", seconds=round(dt, 1))
        except KeyboardInterrupt:
            print("\n  interrupted by user — progress is saved, re-run to resume.")
            raise
        except Exception as e:
            dt = time.time() - t0
            failed.append(label)
            print(f"   ✗ FAILED after {fmt_dur(dt)}: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            prog.record(phase=phase, unit=label, status="failed",
                        seconds=round(dt, 1), error=str(e)[:200])
            # continue to the next unit

    tot = time.time() - t_phase
    print(f"\n{'-'*72}\n{phase} complete: {len(run_times)} ok · {len(failed)} failed · "
          f"{skipped} skipped · elapsed {fmt_dur(tot)}")
    if failed:
        print("  failures: " + ", ".join(failed))
    return {"run": len(run_times), "skipped": skipped,
            "failed": len(failed), "failures": failed}


# ───────────────────────── cycle phase ─────────────────────────────────
def cycle_phase(states, modes, models, target_dt, prog):
    """Run daily cycles for every TRAINED (state, mode) until its cursor reaches
    target_dt (inclusive). Per-cycle global ETA. Skips untrained states and any
    already past the target. Resumable: each state continues from its saved cursor."""
    import live_engine as LE
    import paths

    target_iso = target_dt.strftime("%Y-%m-%d")
    start_iso = getattr(LE, "START_DATE", "2026-01-01")
    if target_iso < start_iso:
        print(f"\n  ⚠  target {target_iso} is BEFORE the data start ({start_iso}). "
              f"Cursors begin at {start_iso}, so there is nothing earlier to cycle to.")

    # build the work list: trained units with how many cycles each needs
    units = []          # (label, city, mode, n_cycles)
    skipped_untrained = skipped_past = 0
    for s in states:
        for m in modes:
            if not paths.metadata_path(s, m).exists():
                skipped_untrained += 1
                continue
            cur = LE._load_state(m, s).get("cursor", start_iso)
            try:
                cur_dt = datetime.strptime(cur, "%Y-%m-%d")
            except Exception:
                cur_dt = datetime.strptime(start_iso, "%Y-%m-%d")
            n = (target_dt - cur_dt).days + 1
            if n <= 0:
                skipped_past += 1
                continue
            units.append((f"{s}/{m}", s, m, n))

    total_cycles = sum(u[3] for u in units)
    print(f"\n{'='*72}\nPHASE: cycle  →  until {target_iso}\n"
          f"  {len(units)} trained unit(s) to advance · {total_cycles} cycle-days total · "
          f"{skipped_untrained} untrained skipped · {skipped_past} already at/past target\n{'='*72}")
    if total_cycles == 0:
        print("  nothing to do.")
        return {"run": 0, "skipped": skipped_untrained + skipped_past, "failed": 0, "failures": []}

    done = 0
    cyc_times = []
    failed = []
    t_phase = time.time()
    for label, s, m, n in units:
        print(f"\n→ {label}: advancing to {target_iso}")
        print(sysline(), flush=True)
        guard = 0
        while guard <= n + 2:
            cur = LE._load_state(m, s).get("cursor", start_iso)
            if cur > target_iso:
                break
            t0 = time.time()
            try:
                for ev in LE.run_cycle(models=models, mode=m, city=s):
                    if ev.get("type") == "error":
                        raise RuntimeError(ev.get("message", "cycle error"))
                dt = time.time() - t0
                cyc_times.append(dt); done += 1; guard += 1
                avg = sum(cyc_times) / len(cyc_times)
                remaining = total_cycles - done
                eta = avg * remaining
                newcur = LE._load_state(m, s).get("cursor", cur)
                print(f"   [{done}/{total_cycles}] {label} cursor→{newcur} · "
                      f"avg {fmt_dur(avg)} · ETA {fmt_dur(eta)} · finish ~{clock_day(datetime.now()+timedelta(seconds=eta))}",
                      flush=True)
            except KeyboardInterrupt:
                print("\n  interrupted — cursor is saved, re-run to resume.")
                raise
            except Exception as e:
                failed.append(f"{label}@{cur}")
                print(f"   ✗ FAILED at cursor {cur}: {type(e).__name__}: {e}", flush=True)
                import traceback; traceback.print_exc()
                prog.record(phase="cycle", unit=label, status="failed", cursor=cur, error=str(e)[:200])
                break   # stop this unit, move to next
        else:
            print(f"   (safety cap hit for {label})")
        prog.record(phase="cycle", unit=label, status="done",
                    cursor=LE._load_state(m, s).get("cursor"))

    tot = time.time() - t_phase
    print(f"\n{'-'*72}\ncycle complete: {done} cycle-days · {len(failed)} failed · elapsed {fmt_dur(tot)}")
    if failed:
        print("  failures: " + ", ".join(failed[:10]))
    return {"run": done, "skipped": skipped_untrained + skipped_past,
            "failed": len(failed), "failures": failed}


# ───────────────────────── main ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Headless download/tune/train pipeline (sequential).")
    ap.add_argument("command", choices=["download", "tune", "train", "all", "cycle"])
    ap.add_argument("date", nargs="?", default=None,
                    help="for `cycle`: target date to cycle until (DD/MM/YYYY or YYYY-MM-DD)")
    ap.add_argument("--states", default="all",
                    help="comma list of state keys, or 'all' (default)")
    ap.add_argument("--modes", default="all",
                    help="comma list: multivariate,univariate (default both)")
    ap.add_argument("--models", default="all",
                    help="comma list of model names (default: all trainable)")
    ap.add_argument("--trials", type=int, default=None,
                    help="Optuna trials for `tune` (default: config N_TRIALS)")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if a unit looks already done")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="force device (default auto = GPU if present)")
    args = ap.parse_args()

    # device selection must happen BEFORE torch is imported anywhere
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # (cuda/auto: leave the environment alone; engine.DEVICE auto-detects)

    import config as C
    import paths  # noqa

    # resolve scope
    all_states = list(C.STATES.keys())
    states = all_states if args.states == "all" else [s.strip() for s in args.states.split(",") if s.strip()]
    bad = [s for s in states if s not in C.STATES]
    if bad:
        print(f"unknown state keys: {bad}\nvalid: {all_states}"); sys.exit(2)

    all_modes = list(getattr(C, "MODES_ENABLED", ["multivariate", "univariate"]))
    modes = all_modes if args.modes == "all" else [m.strip() for m in args.modes.split(",") if m.strip()]

    models = C.MODEL_NAMES if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    targets = C.TARGETS

    # device banner
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gname = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    except Exception:
        dev, gname = "cpu", "CPU"

    print("="*72)
    print(f"  PIPELINE: {args.command}   ·   {clock()}  {datetime.now().strftime('%Y-%m-%d')}")
    print(f"  device : {dev} ({gname})   quantum backend: {getattr(C,'QUANTUM_BACKEND','default.qubit')}")
    print(f"  states : {len(states)}   modes: {modes}   models: {models}")
    print(f"  trials : {args.trials or getattr(__import__('tune'),'N_TRIALS','?')}   force: {args.force}")
    print("="*72)
    print(sysline(), flush=True)

    prog = Progress(args.command)
    t_all = time.time()
    summary = {}

    if args.command == "cycle":
        if not args.date:
            print("cycle needs a target date, e.g.:  python src/pipeline_all.py cycle 10/05/2026")
            sys.exit(2)
        try:
            target_dt = parse_date(args.date)
        except ValueError as e:
            print(str(e)); sys.exit(2)
        print(f"  cycling until: {target_dt.strftime('%Y-%m-%d')}  ({target_dt.strftime('%d %b %Y')})")
        try:
            summary["cycle"] = cycle_phase(states, modes, models, target_dt, prog)
        except KeyboardInterrupt:
            print("\n\nStopped. Re-run to resume from each state's saved cursor.")
            sys.exit(130)
        # fall through to the shared final summary below
    else:
        try:
            if args.command in ("download", "all"):
                units = [(s, (s,)) for s in states]
                summary["download"] = drive("download", units, run_download, download_done, args.force, prog)

            if args.command in ("tune", "all"):
                # require data first
                units = []
                for s in states:
                    for m in modes:
                        units.append((f"{s}/{m}", (s, m, args.trials, models)))
                def _tune_done(s, m, *_): return tune_done(s, m)
                def _tune_run(s, m, trials, mdls):
                    if not download_done(s):
                        raise RuntimeError("no NASA data — run `download` first")
                    return run_tune(s, m, trials, mdls)
                summary["tune"] = drive("tune", units, _tune_run, _tune_done, args.force, prog)

            if args.command in ("train", "all"):
                units = []
                for s in states:
                    for m in modes:
                        units.append((f"{s}/{m}", (s, m, models)))
                def _train_done(s, m, mdls): return train_done(s, m, mdls, targets)
                def _train_run(s, m, mdls):
                    if not download_done(s):
                        raise RuntimeError("no NASA data — run `download` first")
                    return run_train(s, m, mdls)
                summary["train"] = drive("train", units, _train_run, _train_done, args.force, prog)

        except KeyboardInterrupt:
            print("\n\nStopped. Re-run the same command to resume where it left off.")
            sys.exit(130)

    # final summary
    print("\n" + "="*72)
    print(f"  ALL DONE   ·   total elapsed {fmt_dur(time.time()-t_all)}   ·   finished {clock()}")
    for phase, s in summary.items():
        line = f"   {phase:9s}: {s['run']} ok · {s['failed']} failed · {s['skipped']} skipped"
        if s["failures"]:
            line += "   [" + ", ".join(s["failures"][:8]) + ("…" if len(s["failures"]) > 8 else "") + "]"
        print(line)
    print(sysline())
    print(f"  progress log: logs/pipeline_progress.json")
    print("="*72)


if __name__ == "__main__":
    main()