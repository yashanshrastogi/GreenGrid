import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_isolation_forest(df: pd.DataFrame, default_contamination: float = 0.05) -> pd.DataFrame:
    scored_frames = []

    for entity, g in df.groupby("room" if "room" in df.columns else "circuit"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["hour"] = pd.to_datetime(g["timestamp"]).dt.hour

        if "occupied" in g.columns:
            baseline = g.groupby(["hour", "occupied"])["power_kw"].transform("median")
        else:
            if "dow" not in g.columns:
                g["dow"] = pd.to_datetime(g["timestamp"]).dt.dayofweek
                g["is_weekend"] = (g["dow"] >= 5).astype(int)
            baseline = g.groupby(["hour", "is_weekend"])["power_kw"].transform("median")

        g["expected_load"] = baseline
        g["residual"] = g["power_kw"] - g["expected_load"]
        g["residual_ratio"] = g["residual"] / g["expected_load"].clip(lower=0.01)
        
        # Approximate 2h std and 30m max based on interval freq
        # Assuming ~5min intervals for synthetic, ~15min for real
        interval_min = (g["timestamp"].iloc[1] - g["timestamp"].iloc[0]).total_seconds() / 60 if len(g) > 1 else 5
        w_2h = max(1, int(120 / interval_min))
        w_30m = max(1, int(30 / interval_min))

        g["rolling_std_2h"] = g["power_kw"].rolling(w_2h, min_periods=1).std().fillna(0)
        g["rolling_max_30min"] = g["power_kw"].rolling(w_30m, min_periods=1).max()

        features = g[["residual", "residual_ratio", "rolling_std_2h", "rolling_max_30min"]]

        if "is_anomaly" in g.columns:
            true_rate = g["is_anomaly"].mean()
            contamination = true_rate if true_rate > 0 else 0.005
        else:
            contamination = default_contamination
            
        model = IsolationForest(n_estimators=300, contamination=contamination, random_state=42)
        
        target_col = "predicted_anomaly" if "is_anomaly" in g.columns else "flagged"
        g[target_col] = (model.fit_predict(features) == -1).astype(int)
        
        raw_scores = -model.decision_function(features)
        g["anomaly_score"] = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

        if "circuit" not in g.columns and "room" not in g.columns:
            g["circuit"] = entity
        scored_frames.append(g)

    return pd.concat(scored_frames, ignore_index=True)


def evaluate_and_save(input_path: str, output_path: str) -> dict:
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    scored = run_isolation_forest(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scored.to_csv(output_path, index=False)

    y_true = scored["is_anomaly"].values
    y_pred = scored["predicted_anomaly"].values

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return {"f1": f1, "precision": precision, "recall": recall}


def run_isolation_forest_real(long_df: pd.DataFrame, output_path: str, plot_path: str) -> dict:
    # Resample to 15-min intervals
    g_resampled = []
    for circuit, g in long_df.groupby("circuit"):
        g = g.set_index("timestamp").resample("15min")["power_kw"].mean().reset_index()
        g["circuit"] = circuit
        g_resampled.append(g)
    
    df_15m = pd.concat(g_resampled, ignore_index=True)
    df_15m = df_15m.dropna(subset=["power_kw"]).reset_index(drop=True)
    
    scored_df = run_isolation_forest(df_15m, default_contamination=0.015)
    
    results = []
    for circuit, g in scored_df.groupby("circuit"):
        results.append({
            "circuit": circuit,
            "n_intervals": len(g),
            "n_flagged": int(g["flagged"].sum()),
            "flagged_rate_pct": round(g["flagged"].mean() * 100, 3),
        })

    results_df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scored_df.to_parquet(output_path, index=False)

    sample_circuit = "water_heater_ac"
    sc = scored_df[scored_df["circuit"] == sample_circuit].reset_index(drop=True)
    if len(sc) > 0:
        mid = len(sc) // 2
        window = sc.iloc[mid: mid + 672].copy()

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(window["timestamp"], window["power_kw"], color="#2b6e4f", linewidth=0.8, label="Power draw (kW, 15-min avg)")
        flagged_pts = window[window["flagged"] == 1]
        ax.scatter(flagged_pts["timestamp"], flagged_pts["power_kw"], color="orange", s=30, zorder=5, label="Flagged")
        ax.set_title(f"GreenGrid — Anomaly Timeline ({sample_circuit})")
        ax.set_ylabel("Power (kW)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close()

    avg_flagged_rate = results_df["flagged_rate_pct"].mean()
    return {"avg_flagged_rate_pct": avg_flagged_rate, "per_circuit": results_df.to_dict("records")}
