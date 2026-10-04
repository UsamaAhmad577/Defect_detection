"""Grid of false negatives / false positives with Grad-CAM heat maps."""
import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402

from common import SIZE, build_model, eval_tf  # noqa: E402


def make_grid(rows, name, cam, tf, n=8):
    if rows.empty:
        print(f"[{name}] No samples found. Skipping plot.")
        return
    rows = rows.head(n)
    print(f"\n[{name}] {len(rows)} shown:")
    fig, axs = plt.subplots(2, len(rows), figsize=(2.4 * len(rows), 5.2), squeeze=False)
    for k, (_, r) in enumerate(rows.iterrows()):
        img = cv2.cvtColor(cv2.imread(r.path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        x = tf(image=img)["image"].unsqueeze(0)
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]  # evidence for "defective"
        overlay = show_cam_on_image((img / 255.0).astype(np.float32), heat, use_rgb=True)
        axs[0][k].imshow(img)
        axs[0][k].set_title(f"p_def={r.p_defective:.3f}", fontsize=8)
        axs[1][k].imshow(overlay)
        for a in axs[:, k]:
            a.axis("off")
        print(f"  p_defective={r.p_defective:.4f}  {r.path}")
    fig.suptitle(name)
    plt.tight_layout()
    plt.savefig(f"reports/{name}.png", dpi=130)
    plt.close(fig)


def main():
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load("models/best.pt", map_location="cpu"))
    model.eval()
    # If Grad-CAM complains about the layer, try model.conv_head instead.
    cam = GradCAM(model=model, target_layers=[model.blocks[-1]])
    tf = eval_tf()

    errs = pd.read_csv("reports/errors_test.csv")
    fn = errs[errs.label == 1].sort_values("p_defective")                    # most-missed first
    fp = errs[errs.label == 0].sort_values("p_defective", ascending=False)   # most-confident false alarms first
    print(f"False negatives: {len(fn)} | false positives: {len(fp)}")
    make_grid(fn, "false_negatives", cam, tf)
    make_grid(fp, "false_positives", cam, tf)


if __name__ == "__main__":
    main()
