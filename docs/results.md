# Held-out research-demo results

Protocol v1.2; fixed checkpoints from train-v2; 477 test images, 282 recorded patient IDs, 234 patient/reused-lesion/exact-duplicate components. No test-based model selection or fitting.

| Method | Mean macro-F1 across three seeds | 95% paired-component resampling interval |
|---|---:|---|
| clinical_linear | 0.6439 | 0.5675–0.7019 |
| image_linear | 0.4784 | 0.4250–0.5241 |
| concat_linear | 0.6454 | 0.5827–0.6943 |
| metablock | 0.6809 | 0.6078–0.7336 |
| healnet | 0.6669 | 0.5999–0.7224 |

The prespecified metadata-addition contrast (concat-linear minus image-linear) is **0.1669**, with a 200-replicate paired component-bootstrap interval **[0.1206, 0.2137]**. This is conditional on this split, fixed feature choices, training recipe and fitted models. It is not clinical validation.

Clinical-only macro-F1 is close to simple fusion, and those baselines remained budget-capped. This experiment does not establish that photographs add value over a fully optimized clinical-only model. MetaBlock and HEALNet comparisons are exploratory; their numerical ordering is not a proven architecture advantage.

HEALNet rapidly overfit after its best inner-tuning epoch. All three runs restore epoch3 rather than publishing the final overfit weights. The three clinical-only fits restore epochs95/99/99 after reaching the100-epoch cap; those cannot be called converged.

Test class support, in protocol order ACK/BCC/MEL/NEV/SCC/SEK: **[155, 174, 14, 46, 50, 38]**. Minority-class uncertainty remains substantial.

The results JSON uses the legacy shorthand "patient-plus-exact-duplicate component" in its unit string. The actual IDs used for splitting and resampling also union ambiguous reused lesion IDs; the protocol, component audit and independent review record this full definition.

Training: 15 fits, 234.5 seconds in the worker. No cloud training calls or new cloud resources. The external supervisor sampled the worker process tree; this is not a measurement of total machine energy or all unrelated processes.

See `reviews/final-model-review.md` for the independent recomputation, and `docs/export-parity.json` plus `docs/browser-verification.json` for deployment checks when completed.
