import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
UCI_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "household_power_consumption.txt")
UCI_PARQUET_PATH = os.path.join(PROCESSED_DIR, "uci_circuits_long.parquet")

RNG = np.random.default_rng(42)

ROOMS = [
    "LH-101", "LH-102", "LH-201", "LH-202",
    "Lab-CS1", "Lab-CS2", "Staff-Room", "Library",
]

BASE_LOADS = {
    "LH-101": 1.8, "LH-102": 1.9, "LH-201": 2.1, "LH-202": 2.0,
    "Lab-CS1": 4.5, "Lab-CS2": 4.8, "Staff-Room": 1.2, "Library": 2.3,
}

SCHEDULES = {
    "LH-101": [(8, 11, [0, 1, 2, 3, 4]), (14, 17, [0, 2, 4])],
    "LH-102": [(9, 12, [0, 1, 3, 4]), (15, 18, [1, 3])],
    "LH-201": [(8, 10, [0, 2, 4]), (13, 16, [1, 3, 4])],
    "LH-202": [(10, 13, [0, 1, 2, 3, 4])],
    "Lab-CS1": [(9, 13, [0, 1, 2, 3, 4]), (14, 18, [0, 2, 4])],
    "Lab-CS2": [(10, 14, [1, 3]), (15, 18, [0, 2, 3, 4])],
    "Staff-Room": [(9, 17, [0, 1, 2, 3, 4])],
    "Library": [(8, 20, [0, 1, 2, 3, 4]), (10, 17, [5])],
}

WEEKS = 6
INTERVAL_MINUTES = 5

def is_occupied(room: str, ts: pd.Timestamp) -> bool:
    if ts.dayofweek > 5:
        return False
    for start, end, days in SCHEDULES.get(room, []):
        if ts.dayofweek in days and start <= ts.hour < end:
            return True
    return False

def generate_synthetic_data(output_path: str = None) -> pd.DataFrame:
    start = pd.Timestamp("2024-03-04 00:00:00")
    end = start + pd.Timedelta(weeks=WEEKS)
    index = pd.date_range(start, end, freq=f"{INTERVAL_MINUTES}min", inclusive="left")

    rows = []
    anomaly_count = 0

    for room in ROOMS:
        base = BASE_LOADS[room]
        for ts in index:
            occupied = is_occupied(room, ts)
            
            if occupied:
                load = base * RNG.uniform(0.85, 1.15)
                load += RNG.normal(0, 0.06 * base)
                load += 0.05 * base * np.sin(2 * np.pi * ts.minute / 60)
            else:
                load = base * RNG.uniform(0.08, 0.18)
                load += RNG.normal(0, 0.02 * base)

            load = max(load, 0.0)
            anomaly_type = "none"

            if not occupied and RNG.random() < 0.004:
                load += base * RNG.uniform(0.5, 0.9)
                anomaly_type = "idle_load"
                anomaly_count += 1
            elif RNG.random() < 0.0015:
                load += base * RNG.uniform(1.2, 2.5)
                anomaly_type = "spike"
                anomaly_count += 1

            rows.append({
                "timestamp": ts,
                "room": room,
                "power_kw": round(load, 4),
                "occupied": int(occupied),
                "anomaly_type": anomaly_type,
                "is_anomaly": int(anomaly_type != "none"),
            })

    df = pd.DataFrame(rows)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
    return df

def load_uci_data() -> pd.DataFrame:
    if os.path.exists(UCI_PARQUET_PATH):
        return pd.read_parquet(UCI_PARQUET_PATH)

    if os.path.exists(UCI_RAW_PATH):
        df = pd.read_csv(UCI_RAW_PATH, sep=";", na_values=["?"], low_memory=False)
        df["timestamp"] = pd.to_datetime(
            df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
        )
        df = df.drop(columns=["Date", "Time"])

        numeric_cols = [
            "Global_active_power", "Global_reactive_power", "Voltage",
            "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        ]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=numeric_cols).reset_index(drop=True)

        circuit_map = {
            "Sub_metering_1": "kitchen",
            "Sub_metering_2": "laundry",
            "Sub_metering_3": "water_heater_ac",
        }
        long_frames = []
        for col, name in circuit_map.items():
            sub = df[["timestamp", col]].rename(columns={col: "power_wh"})
            sub["circuit"] = name
            sub["power_kw"] = sub["power_wh"] * 60 / 1000.0
            long_frames.append(sub)

        long_df = pd.concat(long_frames, ignore_index=True)
        long_df["hour"] = long_df["timestamp"].dt.hour
        long_df["dow"] = long_df["timestamp"].dt.dayofweek
        long_df["is_weekend"] = (long_df["dow"] >= 5).astype(int)
    else:
        raise FileNotFoundError(f"UCI raw file not found at {UCI_RAW_PATH}.")

    os.makedirs(os.path.dirname(UCI_PARQUET_PATH), exist_ok=True)
    long_df.to_parquet(UCI_PARQUET_PATH, index=False)
    return long_df
