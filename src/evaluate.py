"""Pick the threshold on VALIDATION, then evaluate on TEST."""
import json
import os

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             classification_report, confusion_matrix, roc_auc_score)
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import DefectDS, build_model, eval_tf, predict_probs  # noqa: E402
from threshold import pick_threshold  # noqa: E402

TARGET_RECALL = 0.95
NAMES = ["normal", "defective"]


def main():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv("data/splits.csv")

    model = build_model(pretrained=False).to(dev)
    model.load_state_dict(torch.load("models/best.pt", map_location=dev))

    def probs_for(split):
        d = df[df.split == split].reset_index(drop=True)
        dl = DataLoader(DefectDS(d, eval_tf()), batch_size=64, num_workers=2)
        p, y = predict_probs(model, dl, dev)
        return d, p, y

    dv, pv, yv = probs_for("val")
    dt, pt, yt = probs_for("test")
    print(f"val: {int((yv == 1).sum())} defective / {int((yv == 0).sum())} normal | "
          f"test: {int((yt == 1).sum())} defective / {int((yt == 0).sum())} normal")

    # ---- 1) threshold from VALIDATION only ----
    threshold, info = pick_threshold(yv, pv, TARGET_RECALL)
    print(f"\nChosen threshold: {threshold:.6f}  ({info['method']})")
    print("val gap info:", info)
    print("Validation confusion matrix at this threshold:\n",
          confusion_matrix(yv, (pv >= threshold).astype(int)))

    # ---- 2) TEST evaluation ----
    pred = (pt >= threshold).astype(int)
    print("\n=== TEST RESULTS ===")
    print(classification_report(yt, pred, target_names=NAMES, digits=4))
    cm = confusion_matrix(yt, pred)
    print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)
    pr_auc = average_precision_score(yt, pt)
    roc_auc = roc_auc_score(yt, pt)
    print(f"PR-AUC: {pr_auc:.4f}  ROC-AUC: {roc_auc:.4f}")
    print("\nFor reference only (never choose a threshold from test data), at 0.5:\n",
          confusion_matrix(yt, (pt >= 0.5).astype(int)))

    # ---- 3) bootstrap confidence intervals ----
    rng = np.random.default_rng(0)
    rs, ps = [], []
    for _ in range(1000):
        i = rng.integers(0, len(yt), len(yt))
        tp = ((pred[i] == 1) & (yt[i] == 1)).sum()
        rs.append(tp / max(1, (yt[i] == 1).sum()))
        ps.append(tp / max(1, (pred[i] == 1).sum()))
    rec_ci = np.percentile(rs, [2.5, 97.5])
    prec_ci = np.percentile(ps, [2.5, 97.5])
    print("Defect recall 95% CI:", rec_ci, "| precision 95% CI:", prec_ci)

    # ---- 4) save artifacts ----
    ConfusionMatrixDisplay(cm, display_labels=NAMES).plot()
    plt.title(f"Test confusion matrix (threshold={threshold:.4f})")
    plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches="tight")

    dt["p_defective"] = pt
    dt["pred"] = pred
    dt[dt.pred != dt.label].sort_values("p_defective").to_csv("reports/errors_test.csv", index=False)

    rep = classification_report(yt, pred, target_names=NAMES, output_dict=True)
    json.dump({"threshold": threshold, "method": info["method"],
               "test": {"report": rep, "confusion_matrix": cm.tolist(),
                        "pr_auc": pr_auc, "roc_auc": roc_auc,
                        "recall_ci95": rec_ci.tolist(), "precision_ci95": prec_ci.tolist()}},
              open("reports/metrics.json", "w"), indent=2)
    json.dump({"threshold": threshold, "method": info["method"], "info": info},
              open("models/threshold.json", "w"), indent=2)
    print("\nSaved reports/confusion_matrix.png, reports/errors_test.csv, "
          "reports/metrics.json, models/threshold.json")


if __name__ == "__main__":
    main()
