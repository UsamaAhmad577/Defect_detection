"""Export to ONNX, verify parity with PyTorch, write metadata.json for the API."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common import MEAN, SIZE, STD, build_model


def main():
    Path("models").mkdir(exist_ok=True)
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load("models/best.pt", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, SIZE, SIZE)
    kw = dict(input_names=["image"], output_names=["logits"], opset_version=18)
    try:   # keep weights inside model.onnx when this torch version supports the option
        torch.onnx.export(model, dummy, "models/model.onnx", external_data=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, "models/model.onnx", **kw)

    # Parity check: ONNX Runtime must reproduce PyTorch on several random inputs.
    sess = ort.InferenceSession("models/model.onnx", providers=["CPUExecutionProvider"])
    worst = 0.0
    for _ in range(5):
        x = torch.randn(1, 3, SIZE, SIZE)
        with torch.no_grad():
            ref = model(x).numpy()
        out = sess.run(None, {"image": x.numpy()})[0]
        worst = max(worst, float(np.abs(ref - out).max()))
        assert np.allclose(ref, out, atol=1e-4), "ONNX output differs from PyTorch"
    print(f"ONNX parity OK (max abs diff {worst:.2e})")

    th = json.load(open("models/threshold.json"))
    meta = {"version": "1.0.0", "size": SIZE, "mean": list(MEAN), "std": list(STD),
            "threshold": th["threshold"], "threshold_method": th.get("method", ""),
            "classes": ["normal", "defective"]}
    json.dump(meta, open("models/metadata.json", "w"), indent=2)
    print("Wrote models/metadata.json with threshold", round(th["threshold"], 6))

    files = sorted(p.name for p in Path("models").glob("model.onnx*"))
    print("ONNX files:", files)
    if "model.onnx.data" in files:
        print("NOTE: weights are in model.onnx.data - the Docker image must copy both files.")


if __name__ == "__main__":
    main()
