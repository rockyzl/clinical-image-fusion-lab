# Entry-guide presentation review

Reviewer: `/root/medical_demo_review`, 2026-11-18. **PASS for the narrow documentation/entry-guide release.** This review does not reopen or extend the earlier scientific validation.

Independent Chromium checks opened both the repository HTML guide and its Windows-share copy at 390 px. Both rendered without horizontal overflow, JavaScript errors or POST requests. Guide bytes matched the Windows copy. A targeted scan of changed text found no credential patterns. No training, model inference, holdout analysis, paid LLM call, deployment, or Windows command launch was performed.

The public demo and repository links returned HTTP 200. The existing 12-case browser-inference and export-parity receipts substantiate removing the stale “when completed” wording. Guide button labels match the current UI source. Its image-plus-record explanation and cautious result interpretation agree with the retained results: the primary fusion-versus-image contrast was positive, while clinical-only remained competitive. Scores are not represented as clinically validated probabilities.

`app`, `src`, `runs`, `protocol.json` and the prior final model review have no diff from HEAD. The Windows Internet shortcut points to the public demo. No app or model change is included in this release.

Nonblocking presentation note: a relative HTML link in a GitHub README opens the source-file view; download/open the HTML locally or use the explicit Windows path for the rendered guide. This does not affect either verified local entry point.

## Reviewed files

| File | SHA-256 |
|---|---|
| `README.md` | `527bb8707764659d87e0716fbd39e1cbda35660b3320f9fb843a7fcc63616f71` |
| `docs/results.md` | `e40b9a986e4c088a97e787b5ef6e543d4ba5561f03264e1ff880dfec35c6c2d7` |
| `docs/start-here.zh.html` | `a717f038872317e566a31bea8b87c65ab0a9fb80aa625a5da82cbfb17274ca27` |

Independent browser check receipt: `/tmp/medical-demo-guide-independent-check.json`, SHA-256 `b4cfda8c0cfcb954301438a2d257c1e35eefc7070af7feede4807c507ee3cef1`.
