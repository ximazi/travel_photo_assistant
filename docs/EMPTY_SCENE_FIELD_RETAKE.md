# Empty-scene guided retake: first field check (#20)

This tests whether a person gets a better original photo after using one
silhouette guide. A public empty photograph can test diagnosis and candidate
ranking only. It cannot provide an actual retake, a usable foothold, or a
shutter outcome.

## Capture-to-review dry run (before recruiting or shooting a scored pair)

Choose a single capture path for both arms. Make two disposable scene-only
captures, close the capture screen, then retrieve both original files to
private storage. Confirm they decode at the expected orientation, retain
their own original bytes, and do not depend on a thumbnail or screenshot.
Generate two anonymous review exports with the same documented orientation,
color rendering and resizing; open both in the intended reviewer viewer.
Preserve a private mapping of the two originals to the neutral export IDs.
Do not treat this dry run as photography-quality evidence.

- **First feasibility run:** if extraction from the app is unverified, use the
  same phone and stock camera for both arms, and use a separate screen to
  show the guide only after the unguided shutter and reset check. Mark this
  as a test of the guide, not app-camera integration.
- **App-camera validation:** [Android smoke #15](https://github.com/ximazi/travel_photo_assistant/pull/15)
  verifies file save and decode on a virtual camera but uses app-private
  temporary cache. Require an operator retrieval path on the target device
  and complete the two-file dry run before claiming end-to-end validation.
- Keep review rendering consistent. Prefer identical SDR review exports
  from both arms and keep original files unchanged. A stock camera may
  produce Ultra HDR even when the extension is .jpg; verify consistent
  rendering in the review viewer or mark the case as unsuitable for blind
  color comparison rather than mixing render paths.
  See [CameraX output formats](https://developer.android.com/reference/androidx/camera/core/ImageCapture).

## One same-scene feasibility pair

1. Before shooting, record composition intent, device, starting lens/view,
   and reasonable time/attempt budget for both arms. Agree to score the
   **first completed original** from each arm; a poor completed photo stays
   in the pair. Record failed saves, extra attempts, skipped guidance and
   time separately. Missing originals are failures, not opportunities to pick
   another image. Use consenting subjects and a safe public setting.
2. Take the empty reference frame. Do not reveal the generated silhouette,
   crop, prompt, or any placement clue to the photographer or subject yet.
3. Have them shoot the unguided original. Verify that it was actually saved.
4. Reset to the same starting view (mark the camera position, camera height,
   starting lens/zoom and orientation). Before showing the guide, save one
   more empty frame with the reset view. Compare a fixed near edge and a
   distant landmark selected in the original empty reference; use stable
   features, not people, waves or shadows. Mark the reset `matched` only
   if both remain aligned in the frame at the same starting lens and
   orientation, `drifted` for visible mismatch or known setup change, or
   `uncertain` if a reference or usable landmark is missing. Record the
   visible mismatch and any light or crowd drift separately. If the case
   proceeds with a drifted or uncertain reset, keep it as a flagged
   feasibility case, not a controlled placement comparison. Keep both
   empty frames as operator-only checks, out of blind review.
   Reveal one suggested pose/placement frame only if the standing area is
   confirmed in person. Let them act on or decline it. Save the guided
   original. The guide need not constrain the final shutter settings
   unless the protocol says so; later guide-driven changes are distinct
   from reset drift.
5. Keep untouched originals. Prepare a private clean review pair from
   those originals, with equal preprocessing, no UI, silhouettes, capture
   condition, filenames or EXIF clues. Randomize presentation order; keep
   the capture mapping separate until results are committed. Have at least
   two reviewers record left/right/tie and one visual reason before seeing
   the mapping. A visual model, if authorized, receives only the clean pair
   and intent, with order reversal as an independent check.

Record per pair: anonymous session ID, first completed original for each arm,
guide shown and accepted or not, whether new frame actually matched the guide,
time from start including model wait, extra attempts, save failures, condition
drift, reset status and near/far anchor evidence, reviewer votes and any
model abstention. Record photo-sharing scope
separately; no photo files or private labels belong in GitHub.

The near/far anchoring check is adapted from the
[NPS repeat-photography field method](https://www.nps.gov/articles/000/aps-20-1-6.htm).
It checks starting-frame comparability only; it does not remove practice
effects or prove a better photograph.

## Interpreting results

Blind review controls presentation bias, not capture learning. Do **not**
counterbalance the same team's capture order: after seeing the guide the team
is no longer genuinely unguided. Conversely, the guided arm always going
second can benefit from practice. Count this as feasibility only, not causal
improvement. For outcome validation independently assign previously unexposed
teams at matched scenes, keep equal capture budgets, and report missing captures,
human win/tie/loss and separate model/human agreement. Do not use winner selection
to fill missing or excluded photos.
