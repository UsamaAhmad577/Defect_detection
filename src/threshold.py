"""Decision-threshold selection (validation data only).

Why not "highest threshold with recall >= target"? When validation scores are perfectly
separated and saturated near 0/1, that rule lands on the extreme edge of the validation
defect scores, and a noticeable share of unseen defects fall just below it.
Instead we take the MIDDLE of the gap between the worst normal and the low end of the
defect scores, in logit space, so there is a safety margin on both sides.
"""
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import precision_recall_curve


def pick_threshold(y, p, target_recall=0.95, eps=1e-6):
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        raise ValueError("Need both classes in the validation set to choose a threshold.")

    z = logit(np.clip(p, eps, 1 - eps))        # logit space: 0.9999 vs 0.9999999 are far apart
    pos, neg = z[y == 1], z[y == 0]
    lo = float(neg.max())                      # above this value: no false positives on val
    hi = float(np.quantile(pos, 1 - target_recall))  # below this value: recall >= target on val
    info = {"worst_normal_logit": round(lo, 3), "defect_quantile_logit": round(hi, 3)}

    if lo < hi:                                # clean gap -> use its middle
        info["method"] = "middle of the validation gap (logit space)"
        return float(expit((lo + hi) / 2.0)), info

    # Classes overlap on validation -> maximise F2 (recall weighted twice as much as precision).
    prec, rec, thr = precision_recall_curve(y, p)
    f2 = 5 * prec[:-1] * rec[:-1] / (4 * prec[:-1] + rec[:-1] + 1e-12)
    info["method"] = "max F2 on validation (classes overlap)"
    return float(thr[int(np.argmax(f2))]), info
