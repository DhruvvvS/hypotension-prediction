import json
import os
import sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from data_loader import load_raw, make_windows, FEATURES
from metrics import evaluate, map_rule_report, bootstrap_auprc_gain, ppv_at_sensitivity
from sklearn.metrics import precision_recall_curve


def split_by_patient(data, test_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    a, b = next(gss.split(data, groups=data["case_id"]))
    return data.iloc[a], data.iloc[b]


def main(csv_path=None):
    data = make_windows(load_raw(csv_path))
    train_all, test = split_by_patient(data, 0.2, 42)
    train, val = split_by_patient(train_all, 0.2, 7)

    pos_weight = (train["label"] == 0).sum() / max((train["label"] == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        early_stopping_rounds=30,
    )
    model.fit(train[FEATURES], train["label"],
              eval_set=[(val[FEATURES], val["label"])], verbose=False)

    p_val = model.predict_proba(val[FEATURES])[:, 1]
    p_test = model.predict_proba(test[FEATURES])[:, 1]

    # alert threshold: lowest score that still gives 80% sensitivity on validation
    prec, rec, thr = precision_recall_curve(val["label"], p_val)
    keep = rec[:-1] >= 0.8
    threshold = float(thr[keep].max()) if keep.any() else 0.5

    print("XGBoost :", {k: round(v, 3) for k, v in evaluate(test["label"], p_test).items()})
    print("MAP only:", {k: round(float(v), 3) for k, v in
                        evaluate(test["label"], -test["cur_map"]).items()})
    print("MAP < 72.5 rule:", {k: round(float(v), 3) for k, v in
                               map_rule_report(test["label"].to_numpy(), test["cur_map"].to_numpy()).items()})
    print("PR AUC gain over MAP:", bootstrap_auprc_gain(
        test["label"], p_test, test["cur_map"], test["case_id"]))

    os.makedirs("models", exist_ok=True)
    model.get_booster().save_model("models/xgb.json")
    with open("models/meta.json", "w") as f:
        json.dump({"features": FEATURES, "threshold": threshold}, f)

    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sv = shap.TreeExplainer(model).shap_values(test[FEATURES])
        shap.summary_plot(sv, test[FEATURES], show=False)
        os.makedirs("reports", exist_ok=True)
        plt.tight_layout()
        plt.savefig("reports/shap_summary.png", dpi=150)
    except ImportError:
        print("shap not installed, skipping the summary plot")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)