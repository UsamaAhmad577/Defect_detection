import time, numpy as np, onnxruntime as ort
s = ort.InferenceSession("models/model.onnx", providers=["CPUExecutionProvider"])
x = np.random.randn(1, 3, 256, 256).astype(np.float32)
for _ in range(10): s.run(None, {"image": x})          # warm-up
t = []
for _ in range(200):
    a = time.perf_counter(); s.run(None, {"image": x}); t.append((time.perf_counter() - a) * 1000)
print(f"p50={np.percentile(t,50):.1f}ms p95={np.percentile(t,95):.1f}ms")