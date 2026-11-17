# Independent draft design review

Reviewer: `/root/oncology_model_review`  
Date: 2026-11-17  
Scope: `protocol.json` version `1.0-draft`, pinned MetaBlock/HEALNet source and licenses, official dataset description and archive manifests.  
Decision: **Revise the draft before freezing it. No training, test access, or execution approval is granted by this review.** No diagnostic labels or held-out outcomes were inspected, and no model was trained.

The paired clinical-photograph dataset is appropriate for a small, real image-plus-metadata demonstration. Patient-level grouping, a frozen external image encoder, excluded outcome/procedure fields, simple controls, and a separate test gate are sound choices. This supports learning and implementation evidence; it does not reproduce either upstream paper's performance or establish clinical usefulness.

## Resolve before the protocol is frozen

1. **Make the primary comparison answer the stated question.** HEALNet minus image-linear changes both inputs and architecture/capacity. That contrast cannot isolate the value of adding metadata. The least costly correction is to preregister `concat_linear minus image_linear` as the primary metadata-addition contrast and keep MetaBlock/HEALNet comparisons exploratory. A stronger alternative is a matched image-only HEALNet ablation, with its extra fits and time budget frozen in advance. Do not choose among these comparisons after seeing test results.

2. **Use the split's independence unit for uncertainty.** If exact-RGB duplicates connect different patient IDs, those patients are not independent. Bootstrap the same patient/duplicate connected components used for splitting; include every image in each sampled component with component multiplicity. The paired comparison must share each bootstrap draw across models/seeds. Explicitly describe the metric as image-level: patients with more photographs contribute more observations. A patient split prevents overlap but does not make an image-weighted metric patient-balanced.

3. **Freeze the complete model recipe.** Record reduced HEALNet depth, latent count/dimension, head counts/dimensions, frequency bands, dropout, weight sharing, self-attention count, activation option, classifier head, and both tensor shapes. The pinned implementation's `self_attn, self_ff = layer[-1]` supports exactly one self-attention pair; keep `self_per_cross_attn=1` or separately review a correction. A practical shape contract is clinical `[B,1,D]` with one spatial axis, and image `[B,7,7,512]` with two axes. A zero-spatial-axis clinical tensor is not supported by the current Fourier-position implementation. A single clinical token has a constant position; it does not give tabular features meaningful spatial order. Fix the MetaBlock channel/token layout and the head/pooling after its output as well.

4. **Preregister undefined-metric behavior.** Do not search splits if a rare class is absent. Missing training/validation classes may trigger the stated fail-closed policy. At test unlock, report six-class support and any absent class without resplitting. Specify fixed class order, `zero_division` behavior for F1, balanced-accuracy averaging, and AUROC/log-loss class handling. A bootstrap draw can omit a rare class even if the complete test set has it; record how undefined AUROC is handled and how many replicates remain. Primary macro-F1 behavior must remain fixed, not silently switch to observed classes.

## Data and leakage checks

- Validate unique image IDs, readable/nonempty images, nonmissing patient/lesion IDs, and consistency of each lesion's patient owner. Reconcile the expected metadata/image inventory; never silently train on whichever downloads succeeded. If one lesion ID spans patients, resolve or conservatively join that group before splitting.
- Hash canonical decoded RGB pixels **with dimensions** and documented EXIF-orientation handling. Exact duplicate control does not detect resized/cropped/recompressed near duplicates. Describe that limitation; optionally perform a prespecified, label-blind perceptual-duplicate audit before freezing groups rather than retrospectively choosing a threshold from test performance.
- Build the patient/duplicate union graph, assign component IDs deterministically, and freeze ordered image IDs, group membership and split hashes before outcome-based selection. Verify no component crosses splits and the cache preserves row identity.
- Treat documented missing/sentinel values consistently, including the provider's `fitspatrick` field spelling. Fit numeric imputation/scaling and categorical vocabulary on training rows only. Freeze missing/unknown category representation and domains before training.
- Extract ResNet features with `eval()` and no gradients so BatchNorm statistics remain those of the pinned ImageNet model. Bind encoder weight SHA256, torchvision version, resize/crop/interpolation/normalization, image bytes, ordered rows and cache dtype/shape. Frozen feature extraction may process held-out images without their labels; no dataset-fitted image transform may use test data. Center crops can remove lesion context, so describe this fixed preprocessing limitation.
- The official dataset reports biopsy confirmation for BCC/SCC/MEL and possible dermatologist-consensus labels for other classes. Keep `biopsed` out of predictors and disclose heterogeneous label ascertainment. Symptoms, dimensions and pathology/diagnostic workflow may be contemporaneous with assessment; this is retrospective classification, not a verified pre-biopsy decision point. No causal or population-wide claim follows from withholding the biopsy field.

## Upstream implementation checks

- Removing HEALNet's bare `except: pass` and copying the caller's input list are justified, minimal hardening changes. Preserve upstream commit/file provenance, retain an unmodified copy or reproducible patch, and place a modification notice in the modified Apache-licensed file. Preserve MetaBlock's MIT notice and HEALNet's Apache license; check whether the upstream distribution includes an additional NOTICE file.
- Verify valid-input numerical agreement between upstream and patched HEALNet with identical weights. Test that repeated forwards do not mutate caller inputs; injected attention errors propagate; both modalities receive nonzero gradients and affect logits; wrong batch/channel shapes fail clearly. Pass no shared attention mask unless its shape is valid for both modalities. Do not claim missing-modality robustness merely because upstream advertises it—this complete-pair experiment does not evaluate that behavior.
- MetaBlock uses `BatchNorm1d`; define a deterministic policy for a final singleton training batch, preferably merge it with the preceding batch. Avoid silently dropping images. All models must consume the same eligible training observations and validation/test evaluation must use `eval()`.
- Record per-model trainable parameter counts. A frozen ResNet plus trained fusion head is genuine transfer-learning/fusion work, but not end-to-end CNN fine-tuning. Reduced HEALNet is an adaptation of its authors' module, not a claimed reproduction of the published benchmark.

## Feasibility and later execution gate

The three official archives total about 3.59 GB compressed; account for archives, extracted PNGs, approximately 230 MB of float32 spatial features, copies, optimizer/activation memory, checkpoints, telemetry and a safety margin against the 12 GiB disk budget. Check extraction inventory/uncompressed size and free space before extraction; reject archive path traversal and hash mismatches.

The proposed 30-minute CPU ceiling needs actual timing evidence. A representative native smoke must traverse real preprocessing/cache access, each model, full intended batch, optimizer/backward, validation and checkpoint restoration, including the largest HEALNet configuration and MetaBlock's final-batch behavior. Project all 15 fits and preprocessing/checkpoint overhead conservatively; cap admission using measured wall time and memory rather than assuming early stopping will make the run fit.

Before a long run, bind complete code/vendor/config/data/cache/dependency/command hashes, emit epoch loss/metric and fit identities, and use an independent external heartbeat/resource monitor with hard limits and durable terminal/failure evidence. The budget must cover the actual worker/process tree, not just the monitor. No training authorization follows from this design review; execution requires a later independent, hash-bound review of actual artifacts and a successful smoke. Test access requires its own later review of frozen selected checkpoints.

## Sources and claim boundaries

- [Official PAD-UFES-20 version 1](https://data.mendeley.com/datasets/zr7vgbcyr2/1): paired smartphone clinical photographs and metadata; 2,298 images, 1,641 lesions, 1,373 patients; CC BY 4.0; mixed biopsy/clinical ascertainment. Retain attribution and identify redistributed/resized example images as such.
- Local pinned MetaBlock: `paaatcha/MetaBlock@e283174409f5e5685cb871bbb9e85c26c3a53c65`, MIT.
- Local pinned HEALNet: `konst-int-i/healnet@90459b65a3d4a4ef9fd405671c457b5a1163cc7d`, Apache 2.0.

Safe project framing: **an independent adaptation of published fusion modules for grouped evaluation of smartphone skin-lesion images plus paired clinical metadata**. It does not cover CT/MRI, whole-slide pathology, radiology reporting, hospital delivery, prospective validation or a clinically usable diagnostic device. Keep `scientific_result=false`, `publication_ready=false`, and `clinical_validation=false` until the relevant independent evidence exists.
