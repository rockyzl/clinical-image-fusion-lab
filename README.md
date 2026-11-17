# Clinical Image Fusion Lab

[Open the live demo](https://rockyzl.github.io/clinical-image-fusion-lab/) · [中文学习说明](docs/learn.zh.md) · [Actual evaluation results](docs/results.md)

A small, independent research implementation that asks: **does a patient's clinical record help a model classify their skin-lesion photograph?** It uses real, paired public records, trains prediction heads, compares five methods and runs the exported models in a browser.

This is clinical-photograph research. It is not CT/MRI, whole-slide pathology, clinical advice, or deployment into a hospital. No EHR or clinical staff system is connected.

## What is real

- PAD-UFES-20 v1: 2,298 actual clinical smartphone photographs paired with de-identified patient metadata. The publisher describes1,373 patients and1,641 lesion IDs. Our key-only audit found250 lesion IDs reused across patients (1,891 patient/lesion pairs); ambiguous IDs are conservatively grouped together before splitting.
- Official archive SHA256 validation, exact decoded-RGB duplicate grouping, image/record identity checks and train-only preprocessing.
- A frozen ImageNet ResNet18 image encoder; trained clinical-only, image-only,concatenation, MetaBlock and reduced HEALNet heads. All five models use the same split and three prespecified random seeds.
- Independent execution review, external CPU/memory/heartbeat monitoring, immutable run records, early stopping on an inner tuning partition, restored checkpoints, separately gated validation/test evaluation and paired component bootstrap intervals.
- Real ONNX image and fusion computation in the browser. The Python API uses the same exported artifacts with version/checksum checks, a restricted input schema, latency logging and a persisted local review-state demonstration.

No LLM or agent is needed for the classification step. An agent could explain saved evidence, but it would not replace image-model training or validation. This project adds image and metadata modeling evidence alongside the separate clinical/gene-expression project; the cohorts are not joined.

## Open-source work reused

[MetaBlock](https://github.com/paaatcha/MetaBlock) (MIT) uses clinical metadata to gate image features. [HEALNet](https://github.com/konst-int-i/healnet) (Apache-2.0,NeurIPS 2024) uses a shared latent representation and attention to combine modalities. We pin the authors' source commits, preserve licenses and test the minimal HEALNet hardening changes against the original. Our reduced classification adaptation does **not** reproduce their published benchmark.

[PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1),Pacheco et al. (2020),is CC BY 4.0. Displayed images are resized/center-cropped derivatives and retain original image IDs. See [THIRD_PARTY.md](THIRD_PARTY.md) and [research notes](docs/research-2026-11-17.md).

## Run locally

```bash
uv sync --frozen
uv run uvicorn app.api:app --host 127.0.0.1 --port 8788
```

Open `http://localhost:8788`. The static page also works with any HTTP static server. A direct file:// opening cannot fetch model files correctly. Browser inference downloads the local ONNX/WASM assets on first use, then runs on the visitor's CPU. There is no paid AI API call or standing cloud VM for the static demo.

The local API exposes `/health`, `/v1/model`, `/v1/predict` and a deliberately labeled `/v1/research-review/{id}` simulation. The review identity is a fixed demo role; it is not SSO, clinical authentication or a real clinician's sign-off. Do not expose this local review API as a patient service. Its SQLite audit database is outside source control.

```bash
uv run python scripts/check_api.py
uv run pytest -q
```

A Dockerfile is provided as a packaging recipe; container execution must be verified in an environment with Docker before claiming it was tested.

## Reproduce the experiment

See `protocol.json`, the independent reviews in `reviews/`, and immutable run records in `runs/`. Downloads and trained `.pt` files are excluded from git. Exported ONNX weights are included for an immediately runnable demonstration.

```bash
uv run python -m src.download
uv run python -m src.data
```

Data preparation is fail-closed and will not overwrite an existing prepared directory. It freezes target-free membership before mechanically partitioning source labels. Analytical label access is separate and logged; this is not a claim that the raw source file lacks outcomes.

Training requires an independently authored, hash-bound execution review of a proposed run. `src.run prepare` creates a proposal; `src.supervisor` validates that review and launches the actual worker. A prior supervised real-backend smoke is mandatory. Validation and test require separate review after checkpoint selection. A durable selection-keyed evaluation claim prevents silently reusing the holdout under a new run name.

## Interpretation limits

The population comes from one Brazilian clinical-photo dataset. All malignant classes were biopsy-confirmed; other labels can reflect clinical consensus. Class prevalence is not a representative screening population. Symptoms and measurements may be contemporaneous with clinical assessment; this is retrospective classification, not verified pre-biopsy decision support. The published label and `biopsed` field are excluded from predictors.

Patient/reused-lesion/exact-duplicate connected components never cross partitions. Conservative grouping may join unrelated records. Near duplicates are not guaranteed absent. Macro-F1 weighs categories equally but images within each category, not unique patients. Bootstrap intervals condition on this split and these trained models. A fixed center crop can remove lesion context. There is no external-site,prospective,fairness or clinical-utility validation.

The browser uses seed 17 for every model,chosen in advance. Displayed cases are the first two test image IDs per class,regardless of prediction correctness. Aggregate comparisons average all three seeds. Prediction scores are not calibrated disease probabilities.

The three clinical-only baselines reached the100-epoch cap while inner-tuning loss was still declining. They are budget-capped references,not evidence of fully converged performance. The prespecified primary contrast is concatenation versus image-only. See [pre-holdout interpretation plan](docs/interpretation-plan.md).
