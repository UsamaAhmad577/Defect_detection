# src/train.py
import random, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from common import *

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def run_epochs(model, tr, va, crit, opt, sched, epochs, dev, best, tag):
    for ep in range(epochs):
        model.train(); tot = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tot += loss.item() * len(y)
        if sched: sched.step()
        pv, yv = predict_probs(model, va, dev)
        ap = average_precision_score(yv, pv)
        print(f"[{tag}] ep{ep+1} loss={tot/len(tr.dataset):.4f} val_PR-AUC={ap:.4f}")
        if ap > best["ap"]:
            best["ap"] = ap
            torch.save(model.state_dict(), "models/best.pt")
    return best

if __name__ == "__main__":
    seed_all()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv("data/splits.csv")
    tr_df, va_df = df[df.split == "train"], df[df.split == "val"]
    tr = DataLoader(DefectDS(tr_df, train_tf()), batch_size=32, shuffle=True, num_workers=2)
    va = DataLoader(DefectDS(va_df, eval_tf()), batch_size=64, num_workers=2)

    n0, n1 = (tr_df.label == 0).sum(), (tr_df.label == 1).sum()
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, n0 / n1]).float().to(dev))
    model = build_model().to(dev); best = {"ap": -1}

    # Stage 1: head only
    for p in model.parameters(): p.requires_grad = False
    for p in model.get_classifier().parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    run_epochs(model, tr, va, crit, opt, None, 4, dev, best, "head")

    # Stage 2: fine-tune everything
    for p in model.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
    run_epochs(model, tr, va, crit, opt, sched, 12, dev, best, "full")
    print("best val PR-AUC:", best["ap"])