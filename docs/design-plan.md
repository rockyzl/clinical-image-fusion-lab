# Design: a photo and the record that belongs to it

Audience: Lu learning multimodal AI, and reviewers assessing an independent portfolio implementation. The page's one job is to make a real matched example and its computed prediction understandable before exposing architecture.

Palette: laboratory white `#FAFCFF`, slate ink `#142A3C`, photo-frame blue `#2458A6`, clinical teal `#08796F`, light blue `#EAF1F9`, warm annotation yellow `#F4CF64`.
Type: restrained Georgia display headings; system sans-serif body; monospace only for record/version IDs. No external font dependency.
Layout: a photograph sits beside its own clinical record, joined by one shared record ID. One button runs the model. Three horizontal probability comparisons answer what happened with image alone, record alone, and both. Dataset evaluation sits below the case; architecture is an expandable view.

```
Clinical image fusion lab                         English / 中文
Can the patient's record help the model read the photo?
[choose a public example]
[224px clinical photo] —— same record —— [age / site / symptoms]
                        [Run the models]
[photo only]       [record only]       [photo + record]
What did the held-out evaluation show?  [actual numbers + interval]
How was this built? [expand architecture, sources, training]
```

Signature: the joined photo-and-record view remains one coherent example when switching records; both modalities visibly change together. This prevents the very confusion the user reported in the previous lab.

Critique before implementation: removed a generic dashboard of technology badges and hero statistics. The first screen now explains an actual input pair. Keep probability bars explicitly labeled model scores, never diagnostic confidence. Use no stock photos, generated images, or invented performance. Build the interface around the real model outputs and show empty/loading/error states. Default English, a full Chinese learning view, keyboard focus, mobile single column, no decorative animation.
