import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, log_loss, confusion_matrix


def score(y, probability):
    prediction = probability.argmax(1)
    return {
        "macro_f1": float(f1_score(y, prediction, labels=list(range(6)), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_auroc": float(roc_auc_score(y, probability, labels=list(range(6)), multi_class="ovr", average="macro")) if len(np.unique(y)) == 6 else None,
        "log_loss": float(log_loss(y, probability, labels=list(range(6)))),
        "confusion_matrix": confusion_matrix(y, prediction, labels=list(range(6))).tolist(),
        "class_support": np.bincount(y, minlength=6).tolist(),
    }


def bootstrap(y, probabilities, groups, iterations=200):
    """Resample patient/ambiguous-lesion/exact-duplicate components, paired across fits."""
    rng = np.random.default_rng(771)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    draws = {name: [] for name in probabilities}
    for _ in range(iterations):
        ids = np.concatenate([members[g] for g in rng.choice(unique, len(unique), replace=True)])
        for name, predictions in probabilities.items():
            draws[name].append(float(np.mean([f1_score(y[ids], p[ids].argmax(1), labels=list(range(6)), average="macro", zero_division=0) for p in predictions])))
    diff = np.array(draws["concat_linear"]) - np.array(draws["image_linear"])
    return {"replicates": iterations, "unit": "patient-plus-exact-duplicate component", "macro_f1_intervals": {name: np.quantile(values, [.025, .975]).tolist() for name, values in draws.items()}, "primary_difference": "concat_linear minus image_linear", "primary_difference_interval": np.quantile(diff, [.025, .975]).tolist(), "conditional_on": "this split and fitted models; fixed six-class F1 with zero contribution for absent classes; not population or clinical validation"}
