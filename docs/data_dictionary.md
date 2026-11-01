# Data dictionary

> **Not validated for clinical use. All data in this project is simulated.** No real patients
> were used. Results show that the pipeline works, not how well the model would predict in real
> patients.

## Source

`scripts/make_synthetic_data.py` (seed 42) writes `data/raw/synthetic_map_data.csv`:
600 simulated patients, about 107,000 patient-minutes. About 40% of patients are given one or two
hypotensive episodes (MAP about 50 to 63 mmHg for 2 to 8 minutes). About 70% of episodes are
preceded by a gradual decline over 8 to 15 minutes. This precursor is built into the simulator.

## Raw columns

| Column | Meaning |

|---|---|
| `case_id` | Simulated patient number (0 to 599) |
| `minute` | Minutes since the start of surgery |
| `map` | Mean arterial pressure, mmHg (clipped to 40 to 150) |
| `age` | Age in years (25 to 85), constant per patient |

## Prediction windows (`src/data_loader.py`)

At each minute `t`, the model sees the previous 10 minutes of MAP and predicts the next 15 minutes.

- A window is skipped if its last MAP is already below 65 (nothing to predict) or if any value in it is missing.
- **Label = 1** if the area below 65 mmHg in the next 15 minutes is at least 20 mmHg-minutes, otherwise 0.
  This is not the same as "MAP below 65 for one minute".

## Features (7)

| Feature | Definition |

|---|---|
| `cur_map` | Last MAP in the 10-minute window |
| `mean_map` | Mean MAP over the window |
| `std_map` | Standard deviation of MAP over the window |
| `min_map` | Lowest MAP in the window |
| `slope` | Linear trend of MAP over the window (mmHg per minute) |
| `delta5` | Last MAP minus the MAP 5 minutes earlier |
| `age` | Age in years |

## Training and evaluation

- Split by patient, never by row: about 64% train, 16% validation, 20% test.
- XGBoost with class weighting. The alert threshold is the highest score that still catches 80% of events on the validation set.
- The score is a weighted ranking, not a calibrated probability.
- The baseline is "just watch MAP", scored by `-cur_map`. The 95% interval for the AUPRC gain over it comes from resampling whole patients.

## Limitations

- Simulated patients with a built-in warning signal. Metrics reflect the simulator.
- No drugs, surgery type, heart rate or other real-world signals.
- Not validated for clinical use.
