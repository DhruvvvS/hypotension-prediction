import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def ppv_at_sensitivity(y, score, target=0.8):
    prec, rec, _ = precision_recall_curve(y, score)
    ok = rec >= target
    return float(prec[ok].max()) if ok.any() else 0.0


def evaluate(y, score):
    return {
        "prevalence": float(np.mean(y)),
        "auroc": roc_auc_score(y, score),
        "auprc": average_precision_score(y, score),
        "ppv_at_80_sens": ppv_at_sensitivity(y, score, 0.8),
    }


def map_rule_report(y, cur_map, threshold=72.5):
    # the "just watch MAP" baseline the paper says we have to beat
    pred = cur_map < threshold
    tp = np.sum(pred & (y == 1))
    return {
        "threshold": threshold,
        "sensitivity": tp / max(np.sum(y == 1), 1),
        "specificity": np.sum(~pred & (y == 0)) / max(np.sum(y == 0), 1),
        "ppv": tp / max(pred.sum(), 1),
    }


def bootstrap_auprc_gain(y, model_score, cur_map, groups, n_boot=500, seed=1):
    # resample whole patients, not rows, so windows from one case stay together
    rng = np.random.default_rng(seed)
    y, model_score, groups = map(np.asarray, (y, model_score, groups))
    map_score = -np.asarray(cur_map)  # lower MAP = higher risk
    idx_by_case = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    cases = list(idx_by_case)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(cases, len(cases), replace=True)
        idx = np.concatenate([idx_by_case[c] for c in pick])
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], model_score[idx])
                     - average_precision_score(y[idx], map_score[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_gain": float(np.mean(diffs)), "ci95": (float(lo), float(hi))}