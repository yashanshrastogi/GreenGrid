"""
GreenGrid — Decision Engine
=============================
Tiered alerting + action logic applied to anomaly-scored data.

Thresholds (adjustable via dashboard):
  score > HIGH_THRESHOLD  → AUTO-SHUTOFF (only if room unoccupied)
  ALERT_THRESHOLD < score ≤ HIGH_THRESHOLD → ALERT (notify facilities)
  score ≤ ALERT_THRESHOLD → LOG (record only)

Impact quantification:
  Energy waste  = (power_kw - expected_load) × interval_hours  [kWh]
  Cost saved    = waste_kwh × COST_PER_KWH  [₹]
  CO2e avoided  = waste_kwh × CO2_PER_KWH   [kg CO2e]

Constants based on Indian grid (2024):
  ₹7.5 / kWh  (institutional rate, MSEB tariff)
  0.82 kg CO2e / kWh  (CEA 2023 emission factor)

Data provenance: SYNTHETIC (occupancy-gated shutoff validated here only;
NOT validated on UCI data which has no occupancy sensor).
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Impact constants
COST_PER_KWH = 7.5          # ₹ per kWh
CO2_PER_KWH = 0.82          # kg CO2e per kWh
INTERVAL_HOURS = 5 / 60     # 5-minute intervals

# Default thresholds (overridable via function args / dashboard sliders)
DEFAULT_HIGH = 0.85
DEFAULT_ALERT = 0.72


def apply_decision_engine(
    scored_df: pd.DataFrame,
    high_threshold: float = DEFAULT_HIGH,
    alert_threshold: float = DEFAULT_ALERT,
) -> pd.DataFrame:
    """
    Apply tiered decision logic to scored anomaly DataFrame.
    Input must have columns: anomaly_score, occupied, power_kw, expected_load
    """
    df = scored_df.copy()

    def decide(row):
        score = row["anomaly_score"]
        occupied = bool(row.get("occupied", 1))

        if score > high_threshold:
            if not occupied:
                return "auto-shutoff"
            else:
                return "alert"   # can't shut off an occupied room
        elif score > alert_threshold:
            return "alert"
        else:
            return "log"

    df["action"] = df.apply(decide, axis=1)

    # Waste = excess above expected (clamped to 0 if below baseline)
    df["waste_kwh"] = (
        (df["power_kw"] - df["expected_load"]).clip(lower=0) * INTERVAL_HOURS
    )
    df["cost_saved_inr"] = df["waste_kwh"] * COST_PER_KWH
    df["co2e_avoided_kg"] = df["waste_kwh"] * CO2_PER_KWH

    return df


def run_decision_engine(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    result = apply_decision_engine(df)

    # Summary
    total_flagged = len(result[result["action"] != "log"])
    print(f"\n=== Decision Engine Summary ===")
    print(result["action"].value_counts().to_string())
    print(f"\nTotal flagged (alert + auto-shutoff): {total_flagged:,}")
    print(f"Total waste caught:      {result['waste_kwh'].sum():.2f} kWh")
    print(f"Total cost saved:        Rs. {result['cost_saved_inr'].sum():.2f}")
    print(f"Total CO2e avoided:      {result['co2e_avoided_kg'].sum():.2f} kg")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Decision engine output saved -> {output_path}")

    return result


if __name__ == "__main__":
    scored_in = os.path.join(BASE_DIR, "data", "processed", "greengrid_scored_data_v2.csv")
    dec_out = os.path.join(BASE_DIR, "data", "processed", "greengrid_decision_engine_output.csv")
    run_decision_engine(scored_in, dec_out)
