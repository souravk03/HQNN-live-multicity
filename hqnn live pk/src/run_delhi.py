"""
ONE-COMMAND Delhi pipeline runner.

Steps (each is skippable with a flag so you can re-run pieces):
  1. download   - pull NASA POWER for Delhi (2021-2024), cache to data/
  2. tune       - Optuna hyperparameter search (writes best_hparams.json)
  3. train      - train all 4 models on first 80%, save weights + metadata
  4. backtest   - walk-forward forecast/verify/readjust over last 20%
  5. report     - print running RMSE/MAE per model

Usage:
  python run_delhi.py                 # full pipeline (tune is slow!)
  python run_delhi.py --skip-tune     # use defaults in config.py, no Optuna
  python run_delhi.py --only backtest # just re-run the backtest
  python run_delhi.py --days 20       # backtest only N days (smoke test)

For genuinely LIVE daily running (after backtest looks good) use live_run.py
under cron instead — that one uses the multi-source provider truth.
"""
import sys
import json
import argparse

CITY = "delhi"


def step_download():
    import nasa
    print("\n[1/5] DOWNLOAD")
    nasa.download_city(CITY)


def step_tune():
    print("\n[2/5] TUNE (Optuna — this is the slow part)")
    try:
        import tune
    except ImportError:
        print("  optuna not installed: pip install optuna  — skipping tune.")
        return
    tune.tune_city(CITY)
    print("  NOTE: review models/delhi/best_hparams.json and copy the best")
    print("        seq_len / lr into config.py before the train step for best results.")


def step_train():
    import train_initial
    print("\n[3/5] TRAIN")
    train_initial.train_city(CITY)


def step_backtest(days):
    import backtest
    print("\n[4/5] BACKTEST (forecast -> verify -> readjust, weekly retrain)")
    backtest.run_backtest(CITY, max_days=days, verbose=True)


def step_report():
    from config import VERIFY_DIR, TARGETS, MODEL_NAMES
    print("\n[5/5] REPORT")
    p = VERIFY_DIR / f"{CITY}_metrics.json"
    if not p.exists():
        print("  no metrics yet.")
        return
    metrics = json.load(open(p))
    print(f"\n  Running RMSE per model ({CITY}):")
    header = "  target".ljust(14) + "".join(m.rjust(10) for m in MODEL_NAMES)
    print(header)
    for tgt in TARGETS:
        row = f"  {tgt}".ljust(14)
        for m in MODEL_NAMES:
            v = metrics.get(tgt, {}).get(m, {}).get("RMSE", "—")
            row += str(v).rjust(10)
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tune", action="store_true")
    ap.add_argument("--only", choices=["download", "tune", "train", "backtest", "report"])
    ap.add_argument("--days", type=int, default=None,
                    help="limit backtest to N days (smoke test)")
    args = ap.parse_args()

    if args.only:
        {"download": step_download, "tune": step_tune, "train": step_train,
         "backtest": lambda: step_backtest(args.days), "report": step_report}[args.only]()
        return

    step_download()
    if not args.skip_tune:
        step_tune()
    step_train()
    step_backtest(args.days)
    step_report()
    print("\nDONE. Ledger: forecasts/delhi_forecasts.csv | "
          "Weights: logs/delhi_weight_changes.json")


if __name__ == "__main__":
    main()
