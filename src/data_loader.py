import numpy as np
import pandas as pd

OBS_MIN = 10        # minutes of history the model sees
HORIZON_MIN = 15    # how far ahead we look for an event
MAP_FLOOR = 65      # mmHg, hypotension threshold
AREA_LIMIT = 20     # mmHg*min below the floor that counts as an event (tune after EDA)

FEATURES = ["cur_map", "mean_map", "std_map", "min_map", "slope", "delta5", "age"]


def load_raw(path=None, n_cases=80, seed=0):
    if path:
        return pd.read_csv(path)
    return _fake_cases(n_cases, seed)


def _fake_cases(n_cases, seed):
    # stand-in data so the pipeline runs before the real preprocessing lands
    rng = np.random.default_rng(seed)
    frames = []
    for cid in range(n_cases):
        n = int(rng.integers(120, 260))
        base = rng.normal(88, 8)
        drift = np.cumsum(rng.normal(0, 1.2, n))
        series = base + drift - 0.02 * np.arange(n)
        for _ in range(rng.integers(0, 3)):  # a couple of dips per case
            s = rng.integers(20, n - 25)
            series[s:s + 12] -= rng.uniform(15, 30)
        frames.append(pd.DataFrame({
            "case_id": cid,
            "minute": np.arange(n),
            "map": series + rng.normal(0, 1.5, n),
            "age": int(rng.integers(25, 85)),
        }))
    return pd.concat(frames, ignore_index=True)


def window_features(obs, age):
    # keep this identical to the copy in api/index.py
    x = np.arange(len(obs))
    return {
        "cur_map": obs[-1],
        "mean_map": obs.mean(),
        "std_map": obs.std(),
        "min_map": obs.min(),
        "slope": np.polyfit(x, obs, 1)[0],
        "delta5": obs[-1] - obs[-6],
        "age": age,
    }


def make_windows(df):
    rows = []
    for cid, g in df.sort_values("minute").groupby("case_id"):
        m = g["map"].to_numpy()
        age = g["age"].iloc[0]
        for t in range(OBS_MIN, len(m) - HORIZON_MIN):
            obs = m[t - OBS_MIN:t]
            if obs[-1] < MAP_FLOOR:
                continue  # already hypotensive, nothing to predict
            future = m[t:t + HORIZON_MIN]
            area = np.clip(MAP_FLOOR - future, 0, None).sum()
            rows.append({"case_id": cid, **window_features(obs, age), "label": int(area >= AREA_LIMIT)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = make_windows(load_raw())
    print(data.shape, "event rate:", round(data["label"].mean(), 3))