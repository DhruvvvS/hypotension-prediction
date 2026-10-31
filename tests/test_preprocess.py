import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocess import to_minute_series, MAP_COL, AGE_COL

def fake_case(minute_maps, age=70):
    """One MAP reading per minute (NaN = no reading), as a per-second frame."""
    t = np.arange(len(minute_maps)) * 60
    return pd.DataFrame({"time": t, MAP_COL: minute_maps, AGE_COL: age})

def test_sample_file():
    df = pd.read_csv(ROOT / "data" / "sample" / "case_sample.csv")
    out = to_minute_series(df, case_id=1)
    assert (out["age"] == 77).all()
    assert out["map"].dropna().between(20, 200).all()
    assert 150 <= out["map"].notna().sum() <= 200

def test_all_artifacts_gives_empty():
    out = to_minute_series(fake_case([-8.0] * 10 + [300.0] * 10), case_id=1)
    assert out.empty

def test_short_gap_filled_long_gap_left():
    maps = [80.0] * 5 + [np.nan] * 2 + [80.0] * 5 + [np.nan] * 5 + [80.0] * 5
    out = to_minute_series(fake_case(maps), case_id=1)
    assert out["map"].iloc[5:7].notna().all()      # 2-min gap filled
    assert out["map"].iloc[12:17].isna().all()     # 5-min gap untouched

def test_missing_age_raises():
    df = fake_case([80.0] * 10, age=np.nan)
    with pytest.raises(ValueError):
        to_minute_series(df, case_id=1)
    assert (to_minute_series(df, case_id=1, age=60)["age"] == 60).all()