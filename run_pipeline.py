import os
import sys
import argparse
import time
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "models"))

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def banner(text: str):
    print(f"\n{text}")
    print("-" * len(text))


def main(skip_prophet: bool = False):
    results = {}
    t0 = time.time()

    banner("Step 1: Generate Synthetic Campus Data")
    from data_loader import generate_synthetic_data
    synth_path = os.path.join(PROCESSED_DIR, "greengrid_synthetic_data.csv")
    generate_synthetic_data(synth_path)

    banner("Step 2: Isolation Forest (Synthetic Data)")
    from anomaly_detector import evaluate_and_save
    scored_path = os.path.join(PROCESSED_DIR, "greengrid_scored_data_v2.csv")
    if1_metrics = evaluate_and_save(synth_path, scored_path)
    results["synthetic_f1"] = if1_metrics["f1"]
    results["synthetic_precision"] = if1_metrics["precision"]
    results["synthetic_recall"] = if1_metrics["recall"]

    banner("Step 3: Decision Engine (Synthetic Data)")
    from decision_engine import run_decision_engine
    dec_path = os.path.join(PROCESSED_DIR, "greengrid_decision_engine_output.csv")
    run_decision_engine(scored_path, dec_path)

    if not skip_prophet:
        banner("Step 4: Prophet Forecasting (Synthetic Data)")
        try:
            from forecaster import run_prophet_synthetic
            forecast_path = os.path.join(PROCESSED_DIR, "greengrid_synthetic_forecast.csv")
            prophet_metrics = run_prophet_synthetic(synth_path, forecast_path)
            results["synthetic_mae"] = prophet_metrics["overall_mae"]
        except ImportError:
            print("prophet not installed. Skipping.")
            results["synthetic_mae"] = "N/A"
    else:
        results["synthetic_mae"] = "skipped"

    banner("Step 5: Load UCI Household Data")
    from data_loader import load_uci_data
    uci_df = load_uci_data()

    banner("Step 6: Isolation Forest (UCI Data)")
    from anomaly_detector import run_isolation_forest_real
    uci_scored_path = os.path.join(PROCESSED_DIR, "uci_scored_real.parquet")
    plot_path = os.path.join(PROCESSED_DIR, "real_data_anomaly_timeline.png")
    if2_metrics = run_isolation_forest_real(uci_df, uci_scored_path, plot_path)
    results["real_flagged_rate_pct"] = if2_metrics["avg_flagged_rate_pct"]

    if not skip_prophet:
        banner("Step 7: Prophet Forecasting (UCI Data)")
        try:
            from forecaster import run_prophet_real
            uci_forecast_path = os.path.join(PROCESSED_DIR, "uci_prophet_forecast.csv")
            plot2_path = os.path.join(PROCESSED_DIR, "real_data_prophet_forecast.png")
            prophet2_metrics = run_prophet_real(uci_df, uci_forecast_path, plot2_path)
            results["real_mae"] = prophet2_metrics["mae"]
        except ImportError:
            print("prophet not installed. Skipping.")
            results["real_mae"] = "N/A"
    else:
        results["real_mae"] = "skipped"

    elapsed = time.time() - t0
    banner(f"Pipeline Complete in {elapsed:.1f}s")

    print("\n=== Model Validation Summary ===")
    print(f"Isolation Forest F1: {results.get('synthetic_f1', 0):.4f}")
    print(f"Prophet MAE (Synthetic): {results.get('synthetic_mae', 'N/A')}")
    print(f"Prophet MAE (Real): {results.get('real_mae', 'N/A')}")

    print("\nData files saved to data/processed/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GreenGrid Pipeline")
    parser.add_argument("--skip-prophet", action="store_true")
    args = parser.parse_args()
    main(skip_prophet=args.skip_prophet)
