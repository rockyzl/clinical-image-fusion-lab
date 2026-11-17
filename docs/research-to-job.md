# What this project can and cannot substantiate

Target: City of Hope Data Scientist, requisition10036578. The role asks for deep learning, multimodal modeling, medical-image familiarity and ML deployment, alongside LLM applications and established data-science work.

| Requirement | Concrete project evidence | Boundary |
|---|---|---|
| Medical-image modeling | Real public skin-lesion photographs, frozen ResNet18 image features, trained classification heads, image/metadata identity checks | Clinical smartphone photographs only; no CT/MRI, DICOM, segmentation or whole-slide pathology |
| More complex multimodal models | Paired clinical metadata and images; official MetaBlock gating and reduced HEALNet attention adapted with source attribution and parity tests; simple controls | Independent adaptation, not original algorithm development or reproduction of the upstream benchmark |
| Model training and evaluation | Five methods × three seeds; grouped split; train-only preprocessing; inner tuning; frozen checkpoint replay; separate holdout evaluation | Small single-dataset retrospective experiment, no external clinical validation or generalizable efficacy claim |
| Model deployment | Exported ONNX encoder/heads, actual browser computation, versioned local FastAPI inference, model checksums and input validation | Research deployment. No hospital integration, clinical SSO, authenticated clinical reviewer, prospective monitoring or patient-care use |
| LLM/agent applications | Covered by Lu's existing enterprise experience and separate METABRIC evidence assistant | The image classifier is not an LLM or agent; no extra agent is implied |
| Genomics | Separate existing METABRIC clinical/gene-expression prototype | The two cohorts are not joined; this image project does not add raw sequencing expertise |

Resume statements are permitted only after the relevant implementation and checks actually complete. Results and links must come from the final release artifacts. Never convert a prototype into employer experience,clinical deployment or years of hands-on specialist experience.
