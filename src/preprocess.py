import numpy as np
import pandas as pd

MAP_COL = "Solar8000/ART_MBP"
AGE_COL = "Primus/SET_AGE"
MAP_MIN, MAP_MAX = 20, 200   # outside this is line-zeroing / flush artifact
MAX_GAP_MIN = 3              # only fill gaps up to this many minutes


def to_minute_series(df, case_id, age=None):
    # Per-second monitor rows -> one clean MAP value per minute.
    df = df.dropna(subset=["time"])  # exports can end with a blank/padded row
    if age is None:
        ages = df[AGE_COL].dropna()
        if ages.empty:
            raise ValueError(f"case {case_id}: no age given and {AGE_COL} is empty")
        age = float(ages.iloc[0])

    m = df[MAP_COL].where(df[MAP_COL].between(MAP_MIN, MAP_MAX))
    per_min = m.groupby((df["time"] // 60).astype(int)).median().dropna()
    if per_min.empty:
        return pd.DataFrame(columns=["case_id", "minute", "map", "age"])

    s = per_min.reindex(range(per_min.index.min(), per_min.index.max() + 1))
    isna = s.isna()
    run_len = isna.groupby((isna != isna.shift()).cumsum()).transform("sum")
    s = s.interpolate(limit_area="inside").where(~isna | (run_len <= MAX_GAP_MIN))

    # Returns columns case_id, minute, map, age.
    return pd.DataFrame({"case_id": case_id, "minute": s.index, "map": s.to_numpy(), "age": age})