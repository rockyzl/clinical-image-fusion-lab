# Deployment evidence planned before the held-out evaluation

This is a versioned research inference deployment, not deployment into a hospital or a clinical workflow. The same frozen checkpoints will serve a local FastAPI endpoint and a free static browser demo, with no new cloud VM or paid model API.

The browser will run a frozen ResNet18 ONNX image encoder followed by five exported prediction heads. It uses seed17 for all models, fixed before test inspection. Display cases are the first two test image IDs in each of the six source classes, including mistakes; no case is selected for prediction quality. Dataset scores remain the mean of all three seeds, not the displayed seed alone.

Image preprocessing matches the official ImageNet resize256/center224/PIL bilinear pipeline. Public demo PNGs preserve the transformed224×224 pixels; browser normalization uses ImageNet mean/std. Encoder features are quantized float16 then restored float32 to mirror training-cache storage. Python ONNX and browser results must agree with PyTorch checkpoint inference within documented tolerances before release. Report any discrepancies or failed exports rather than serving a cached response as a live calculation.

A local API will reject unknown case/model IDs, expose health/model-version endpoints, log inference identifiers and latency, and require an explicit demo-review action before a result is marked reviewed. It is a local technical simulation, not clinical staff integration. All data is publicly released, de-identified research data; no arbitrary patient uploads are accepted.
