"""Generate a SIMULATED intraoperative MAP dataset (no real patients).
Output columns: case_id, minute, map, age (one row per patient-minute).
"""
import os

import numpy as np
import pandas as pd


def generate_synthetic_dataset(n_patients=600, seed=42):
    np.random.seed(seed)
    records = []
    n_with_events = 0

    for case_id in range(n_patients):
        duration = np.random.randint(120, 240)          # minutes
        minutes = np.arange(duration)
        age = int(np.clip(np.random.normal(60, 15), 25, 85))
        baseline_map = np.random.normal(90, 8)

        # random-walk drift with a slight downward trend, plus measurement noise
        drift = np.cumsum(np.random.normal(0, 0.8, duration)) - 0.05 * minutes
        base = baseline_map + drift + np.random.normal(0, 1.5, duration)
        map_values = base.copy()

        if np.random.rand() < 0.40:
            n_with_events += 1
            for _ in range(np.random.randint(1, 3)):
                start = np.random.randint(10, max(11, duration - 20))
                end = min(start + np.random.randint(2, 9), duration)
                level = np.random.uniform(50, 63)        # MAP during the episode

                # gradual decline before ~70% of episodes (the "warning")
                if np.random.rand() < 0.70:
                    lead = min(np.random.randint(8, 16), start)
                    map_values[start - lead:start] -= np.linspace(0, np.random.uniform(8, 20), lead)

                map_values[start:end] = level + np.random.normal(0, 1.5, end - start)

                # recovery: start halfway back, return to the underlying signal
                rec = min(5, duration - end)
                if rec > 0:
                    w = np.linspace(0.5, 0, rec)
                    map_values[end:end + rec] = base[end:end + rec] * (1 - w) + level * w

        map_values = np.clip(map_values, 40, 150)

        for t in range(duration):
            records.append({"case_id": case_id, "minute": t,
                            "map": round(map_values[t], 1), "age": age})

    return pd.DataFrame(records), n_with_events


if __name__ == "__main__":
    n = 600
    df, n_events = generate_synthetic_dataset(n_patients=n)

    os.makedirs("data/raw", exist_ok=True)
    out = "data/raw/synthetic_map_data.csv"
    df.to_csv(out, index=False)

    low = df[df["map"] < 65]["case_id"].nunique()
    print("SIMULATED data - not real patients")
    print(f"Saved to: {out}")
    print(f"Rows: {len(df):,} | Patients: {df['case_id'].nunique()}")
    print(f"Patients given an injected episode: {n_events} ({n_events / n * 100:.1f}%)")
    print(f"Patients with any MAP < 65: {low} ({low / n * 100:.1f}%)")
    print(f"Mean MAP: {df['map'].mean():.1f} mmHg | Std: {df['map'].std():.1f} mmHg")