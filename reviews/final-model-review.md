# Independent model and bounded deployment review

Reviewer: `/root/oncology_model_review`, through the collaboration tool. Date: 2026-11-17.

**Decision:** approved as an independently reviewed research/portfolio implementation and locally verified inference demonstration. This is not approval of clinical efficacy, a scientific publication, a general architecture ranking, or hospital deployment. `scientific_result=false`, `publication_ready=false`, and `clinical_validation=false` remain unchanged. Public GitHub Pages availability was not verified in this review.

## Evidence and scope

I rehashed the actual source, configuration, dependency and prepared-data inventory; replayed the training/test start, independent approval, exact command, successful terminal journal, checkpoint identity and durable evaluation-claim chain; and checked that the fixed selection remained unchanged after validation. The test run completed cleanly in 28.306 seconds, with no monitor cleanup actions and one recorded analytical test read.

The review used the saved `runs/test-v2/predictions.npz` for labels and probabilities. It did **not** reopen original held-out label files, retrain a model, choose a new checkpoint, or initiate another held-out execution. Its 477 ordered row indices match the frozen test membership, and its component IDs match the source-key table: 282 patient IDs in 234 patient/ambiguous-lesion-ID/exact-RGB components. Identity checks concern this recorded provenance; they do not establish clinical ground truth independently of the public dataset.

All 15 probability arrays have shape 477 × 6, contain finite values within [0, 1], and sum to one within 2.39e-7. All planned model/seed identities and frozen checkpoint hashes match. Training and restoration were checked earlier by independently replaying all 15 checkpoints on the 261 inner-tune images; the maximum selected-loss discrepancy was 1.12e-7.

## Independently reproduced metrics

I recomputed confusion matrices, supports, macro-F1 and balanced accuracy directly from class counts, AUROC using an independent tied-rank formulation, and log loss directly from probabilities. All 15 confusion matrices, supports, F1 and balanced-accuracy values match exactly. Maximum AUROC and log-loss differences were 7.60e-9 and 5.42e-8, consistent with floating-point arithmetic.

The bootstrap replay used a separate weighted-confusion implementation: 200 shared draws of whole connected components, RNG seed 771, averaging the metric across all three prespecified seeds. It did not average probabilities into an ensemble. All model intervals and the primary paired-difference interval agree within 1e-12. Details are preserved in `reviews/final-metric-replay.json`.

| Method | Mean test macro-F1 | 95% component-bootstrap interval |
|---|---:|---:|
| Clinical linear, budget-capped | 0.643856 | 0.567523–0.701874 |
| Image linear | 0.478442 | 0.424980–0.524150 |
| Concatenation linear | 0.645357 | 0.582733–0.694301 |
| MetaBlock | 0.680942 | 0.607794–0.733567 |
| Reduced HEALNet | 0.666850 | 0.599909–0.722395 |

The prespecified **concat-linear minus image-linear** difference is **0.166915 macro-F1**, with interval **[0.120625, 0.213693]**. Safe interpretation: adding these clinical fields improved this fixed image-linear recipe on this held-out split. The interval is conditional on this dataset, split, feature choices and fitted models; 200 replicates provide limited precision for tail quantiles. It does not establish transportability or clinical utility.

## Required interpretation limits

- The three clinical-only models hit the 100-epoch cap while inner-tune loss was still declining. Their selected epochs were 95/99/99. They are budget-capped, not demonstrated converged. Clinical-only and concatenation performance are close; the experiment does not establish that images add value over an optimized clinical model.
- Both primary controls satisfied the prespecified 15-epoch patience rule. Image-linear selected epochs 10/10/14; concatenation selected 18/18/18. The capped clinical-only model therefore does not invalidate the stated metadata-addition contrast.
- HEALNet selected epoch 3 for all seeds and subsequently overfit sharply. The saved/restored epoch-3 weights were correctly evaluated. MetaBlock and HEALNet remain exploratory comparisons of these reduced fixed recipes; their numerical ordering is not proof that one general architecture is superior.
- This is one retrospective Brazilian clinical-photograph dataset, not CT/MRI, DICOM, whole-slide pathology, external-site validation or prospective diagnosis. The image encoder is frozen ImageNet ResNet18; the trained parts are prediction/fusion heads, not end-to-end image-encoder fine-tuning.
- Labels have mixed ascertainment. Clinical symptoms and measurements may reflect contemporaneous assessment. Excluding the explicit diagnosis and biopsy indicator does not establish a verified pre-diagnostic use case.
- Conservatively grouping reused lesion IDs can over-group unrelated patients; exact-RGB grouping does not guarantee removal of near duplicates. Test class support is [155, 174, 14, 46, 50, 38] in ACK/BCC/MEL/NEV/SCC/SEK order. Minority-class uncertainty remains substantial.
- F1 is image-level within classes; uncertainty resamples components. Scores are uncalibrated class scores, not validated disease probabilities. The old result field `patient-plus-exact-duplicate component` is incomplete wording: actual component IDs also include ambiguous shared lesion IDs. The public results document and UI correctly explain the full grouping rule.

## Local inference deployment evidence

I inspected the export, browser-verification and local API code and artifacts. The encoder export loads the original prepared encoder weights; all five exported heads bind to their seed-17 frozen checkpoints. I independently rehashed all six ONNX models and checked the manifest's embedded test results against the approved results file.

The 12 displayed records are exactly the first two frozen test IDs in each class, as declared before test inspection. Their labels do not enter inference. Their clinical vectors match the prepared features, and all released 224 × 224 image pixels match the prescribed resize/center-crop of the original paired source images. The manifest supplies inputs and aggregate evaluation results, without cached per-case prediction arrays.

I independently compared native ONNX reference outputs with saved PyTorch test predictions across all 12 cases and five heads: maximum absolute probability difference **8.23e-6**. The inspected Chromium verification script actually invokes the encoder and all five heads; its saved 12-case report shows maximum browser/native-ONNX difference **9.22e-6**, no page errors, successful language switching and no mobile horizontal overflow. I inspected that browser run's evidence rather than claiming to have launched a second browser run myself.

I independently reran `scripts/check_api.py`: all nine checks passed, including actual ONNX inference, restricted inputs, persisted local review state and rejection of a second review. The checked prediction differs from its native ONNX reference by **2.98e-8**. The review role is a fixed local demonstration string, not authentication or a clinician identity. This supports a **local research inference API and browser model execution** claim, not hospital integration, clinical authorization or a publicly operated API. Docker execution and the eventual public hosting URL remain outside this review.

## Safe resume language

Suitable under an explicitly labeled **Independent research prototype** project section:

- Adapted the authors' MetaBlock and HEALNet modules to explore fusion of public skin-lesion photographs and paired clinical metadata; compared five models across three fixed seeds using patient/lesion/duplicate-grouped splits and held-out evaluation.
- Exported a frozen ResNet18 encoder and trained prediction heads to ONNX for browser inference; checked prediction agreement against PyTorch and built a versioned local FastAPI demonstration with input checks and persisted review state.

A numeric performance bullet is optional and less useful than the implementation evidence. If included, state the comparator, retrospective dataset and fixed split; do not describe macro-F1 as diagnostic accuracy, imply an original fusion algorithm, claim clinical employment experience, or claim a hospital deployment. The static demo can be called publicly available only after its release URL is actually verified.

## Bound evidence

- Frozen selection SHA256: `12fd0dcbc8975120ebd63a332e3e017886fd9a62fbd21d4019cac0bbd86c045c`
- Test results SHA256: `c7c2085f587e24b677b9c9fec2e33e5d94bc99924aa2a2526191d8f0b3087917`
- Saved test predictions SHA256: `967bc4ad1dff0f6864aeddaa5bc9c0742d4384b51105cecad5775c4e293968ac`
- Test start SHA256: `bea6fcc2026dd90d3756e50dac0019ddf3da72dc3dacc6812ad7ca7b4fd130ee`
- Reviewed release manifest SHA256: `81f3863b9280a0baaf1f0b819fd2d0d9070d2c9c5234b9053cf643f9c23d05b7`
- Export parity evidence SHA256: `7c137c849d6143d3754d2f3af39b13e6e6d9ec478d379adb5363003e1fc9f3c1`
- Browser verification evidence SHA256: `af6be6fc03c6d1b204cecbe1fcb4689d4d141fdfd8987f155375cd2e3fbe394e`

No blocking model or local inference correctness finding remains for the bounded claims above. Changes to published links alone do not change the model evaluation; changes to models, inputs, selection policy or reported results require a new review.
