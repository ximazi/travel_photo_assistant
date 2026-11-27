# 开发集 DEV-01：峡谷日落，两模型首次对比（2026-11-27）

Refs #31 and Draft #32. This is one development pair for debugging the judge and scoring rule, not a held-out human benchmark or a controlled guided retake.

## Inputs fixed before review

- A = [Photographing the River](https://npgallery.nps.gov/AssetDetail/d47518f4-6c75-48c2-815f-79ebcc65a259); B = [Visitor viewing sunset](https://npgallery.nps.gov/AssetDetail/3339ed1d-3857-41bb-bad1-ffda772a6dd3). NPS / Victoria Stauffenberg. Both NPS asset records explicitly mark public domain. The shots were taken 80 seconds apart at the same site; visible people have no recognizable faces in the transmitted frames. Same camera and focal length in source records, but exposure, composition and visible clothing/actions differ; no same-person or retake assumption.
- JPEGs derived from originals by orientation normalization, metadata stripping, same long side (1200 px), and no crop. A 1200×800 SHA256 `f189c1bad23fee5bfaf9df32994a26545568f4cf3db354f2c7bcaa33d9c45b9d`; B 1200×693 `dc8beed5f8dd04d9a9f881c7101324e4e10940fb19e25bce5e2abb3276972bd8`. Source SHA256 A `a46da36bcf711a4de5b7a515e12b4d5d4533d6d1237fa396babf0109dbdcee2c`; source B `3e4060402dfc3149f06547ac683baeb21099cc224c5e19835c63a715053ee914`.
- Same portrait-review-v0.2 prompt SHA256 `a7f05fcd1e59355009ddf6ab580932e12db99ed8effc9785587f62b4d7eaa9f6`. Intent: an environmental travel photograph of a traveler experiencing a canyon at sunset; a visible real traveler and readable canyon depth are hard requirements. Neither model sees captions, label hints, or the other model's vote. Same prepared bytes; model-side resizing may differ, original aspects preserved. Temperature 0 with JSON mode and 3000 max output tokens. Ollama context 8192. Qwen forward first, Gemini reverse first. Two requests per model, no retries. Qwen local only; Gemini configured no-billing project route.
- Pre-model visual check by assistant, not human consensus: A's traveler taking a picture is larger in the lower right; one shoe near the edge, foreground stone occupies more space, and the river is clear. B shows a smaller traveler, visible sun and clearer sky balance; the river is still readable. B is the operator's visual preference for this specific sunset brief; A may be preferable for a photography-action brief. The ledge shown in B is **not** a place recommendation or evidence of safe access.

## Results normalized to original source order

| Judge | A/B presentation | B/A presentation | admissible result |
|---|---|---|---|
| Gemini 3.1 Flash-Lite | A 80; B 97.5; beauty B, intent tie; 15.297 s | A 80; B 100; beauty B, intent tie; 26.765 s | 2/2 valid: prefers original B for beauty, equal intent |
| Local Qwen3-VL 4B | A 100; B 100; raw verdict first for beauty and intent; 11.468 s | A 100; B 100; raw verdict first again for both; 9.032 s | 0/2 valid: score/verdict contradictions; **no vote** |

Reported Qwen totals are diagnostic recomputations, not accepted ratings. Qwen did recognize visual details independently in at least the forward output (camera use vs sunset view), so this is not evidence of a missing image. It failed to differentiate scores and asserted a different source winner when switched. Gemini's raw reason mentions the clearer hierarchy and sunset on B; it also gave B 97.5 vs 100 for the same pixels across orders (technical 3→4), so a stable winner is not stable absolute scoring. Both images fulfill the two hard requirements according to Gemini; this does not mean they express the sunset equally well.

## What changes next

1. Do not admit Qwen 4B as an automatic judge based on transport or this pilot; keep both raw invalid returns and show invalids in the denominators. Do not average Qwen's all-4 scores with Gemini.
2. Separate two experiments on development samples: (a) make deterministic total/tie arithmetic authoritative while retaining any stated winner as a diagnostic, and (b) sharpen the 3→4 boundary with named, visible advantages and evidence of meaningful compromises; unearned all-4 remains a substantive error even if arithmetic is fixed.
3. Recheck each experiment across independent model families and diverse dev pairs, including hard intent violations and near ties, before selecting any revision. Add independent blind human votes; do not count operator preference or this report as human labels. No weight changes or quality claims from one pair. No retake, field, or model-human accuracy reported.
