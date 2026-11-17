# Independent executable review before freezing

Reviewer: `/root/oncology_model_review`  
Date: 2026-11-17  
Status: **Changes required before an execution proposal can be approved.** This is read-only code/method review, not an execution approval. No label file was opened and no model or data preparation was executed.

Reviewed: draft protocol, `src/data.py`, `src/run.py`, `src/common.py`, `src/models.py`, `src/metrics.py`, model/data tests, patched and original HEALNet. The supervisor was still under implementation and is not approved by this review.

## Material gate issues

1. **Smoke provenance is not connected to the proposed training recipe.** `proposal(mode="train")` currently accepts a clean resource flag, smoke mode and five trial rows. It does not verify the smoke's start/proposal binding against the current source, protocol, prepared cache, dependency lock, batch shape or exact expected model/seed identities. `verify_review` only rehashes the two accepted smoke artifacts. Consequently an obsolete smoke can certify changed training code or architecture. Before feasibility admission, replay the smoke start, complete journal, terminal/resource status, exact identities and checkpoint metadata/hashes; require the scientifically relevant binding and intended device/batch recipe to match the proposed run. Bind those verified artifacts in the proposal.

2. **Held-out selection eligibility is checked too weakly and too late.** Evaluation proposals currently check only `mode == train` and trial count. `evaluate` opens validation/test labels before verifying every checkpoint's semantic identity, and never compares saved protocol identity to the current protocol. A stale model trained with a different feature encoding can therefore consume held-out labels before a mismatch is detected. Before any protected label read, require:
   - exact ordered or uniquely keyed model/seed identities equal to the complete frozen plan;
   - successful, reconciled training start/journal/terminal/resource evidence;
   - training source/protocol/cache/preprocessing binding consistent with the evaluation recipe;
   - all selected checkpoint physical hashes and model/seed/epoch/start/protocol metadata validated;
   - no failure/partial-run marker and no missing planned trial.
   Reuse one validated selection loader so proposal review and execution cannot diverge. Do not rely on length alone or a caller-supplied `approved_by_threshold` flag.

3. **Test access accounting is only per worker process.** `Store.reads` blocks repeated reads inside one worker, but a second independently named evaluation directory resets it. If the intended contract is a single final test analysis per frozen selection, claim a durable create-exclusive unlock keyed to that selection (or enforce the equivalent in the supervisor). Record analytical access when it happens, including failure paths. A separately reviewed recovery must disclose previous access; it must not silently reset the count. This is a workflow correctness check, not a claim that local JSON creates a cryptographic security boundary.

4. **The future supervisor must validate the complete seam.** Worker return code alone is insufficient. Before accepting completion, reconcile the exact reviewed proposal/command, its one-use output claim, resource monitor, journal chain, planned identities, selected checkpoint metadata and terminal status. If the worker is killed, the supervisor must preserve failure/resource evidence without manufacturing a clean worker terminal. Review direct-worker entry behavior so it cannot accidentally become the documented path around the independent execution gate. Add negative tests for stale smoke, duplicated/missing trials, checkpoint protocol mismatch, failure after output claim and malformed/post-terminal journals, with a sentinel proving held-out labels were not opened.

## Data-preparation and metric corrections

- `ClinicalEncoder.fit` reserves `__UNKNOWN__` but only reserves `__MISSING__` if it was observed in fit rows. Consequently previously unseen missingness is encoded as unknown, despite the protocol promising distinct missing/unknown categories. Reserve both deterministically in every vocabulary and test a complete fit column followed by separate missing and novel categories. Also avoid adding duplicate reserved vocabulary entries.
- `assign_split` uses hard-coded 0.6/0.8 thresholds while the protocol exposes train/validation/test fractions. Current values agree, but a future protocol edit would silently be ignored. Derive cutoffs from the frozen fractions and validate their sum/ranges.
- Document image orientation explicitly. Both current hashing and extraction ignore EXIF orientation, so they are internally consistent, but the recipe must say so or apply the same canonical orientation in both paths. Exact RGB checks still do not detect near duplicates.
- Validate cached float16 values after casting, scaler finiteness/positive scale, row counts and final extraction cursor before marking preparation complete. The current float32-only finiteness check occurs before float16 storage. This is unlikely to overflow for ordinary ResNet activations but should fail during preparation rather than after labels are consumed.
- The reserved class order and fixed-label macro-F1 are correctly implemented. Balanced accuracy averages classes present in the evaluated sample; AUROC is null when a class is absent. Bootstrap currently covers macro-F1 only, so say that explicitly rather than suggesting AUROC intervals are also produced. Add the primary difference point estimate to the uncertainty record alongside its interval, using the same mean-per-seed definition.
- Clinical images/sample rows are retained and image-level scoring is explicit. Connected-component bootstrap correctly keeps each component's images together and shares draws across models. Keep the image-weighting versus patient-weighting limitation visible.

## Feasibility and recipe completeness

- Smoke uses one training and one tuning batch, whereas full training has a different fit/tune batch ratio. `mean_batch_seconds` merges both costs. Use separately measured training-batch and tuning-batch timings with their real counts, plus checkpoint/monitor overhead and a safety margin. Current 1.8 margin may be conservative, but the component-based projection is auditable and avoids depending on that assumption.
- Before freezing, explicitly record HEALNet `weight_tie_layers=false`, Fourier encoding enabled and final classifier enabled rather than relying only on upstream defaults. Record feature-cache float16, fit-only channel scaling, the exact ResNet transforms and extraction batch size in provenance. The implementation is a reduced adapted HEALNet, not the upstream paper's trained model.
- Frozen encoder evaluation/no-gradient mode, fitting both clinical preprocessing and image-channel scaling only on the `fit` partition, and withholding tune/outer labels from those transforms are correct. The hash rule for an inner tuning subset is target-free and preserves components.
- The balanced-batch helper preserves every row for normal partitions and avoids a singleton tail; the existing tests cover useful boundary cases. The existing HEALNet parity, input nonmutation, dual-modality gradient and fail-closed tests are useful. They do not substitute for the supervised native smoke of the full executable path.

No additional modeling search or dataset change is recommended. Fix the execution gates and small preprocessing/recipe ambiguities, freeze the final artifacts, and then request the separate instrumented-smoke review.
