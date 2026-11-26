# Manual real-pixel smoke: empty-scene shortlist (#20)

**Scope:** geometry, raster overlay and clean crop export on real pixels with
manually supplied labels and proposals. Not observer accuracy, blind model
ranking, standing safety or reshoot quality. No network model requests.

## F20-SMOKE-01: stone-paved old-town lane

Input: existing `assets/examples/photos/1.png`, Git blob
`6621c4c7648395c7d808e0040bbe09c32de404e7`, 909 × 637 RGBA. Visual inspection:
there is no clear intended subject; one tiny distant shape on the right is
ambiguous. Exclude from blind emptiness metrics. Repository sample external
use rights are not established; do not send its pixels to a provider.

Manual upright normalized labels:
- Possible ground: foreground paving `[.365,.65,.735,.92]`.
- Important anchors: red lantern `[.376,.399,.406,.458]`, far roof
  `[.468,.252,.595,.375]`.
- Obstacle cue: right yellow pole `[.787,.350,.856,.747]`.
- Subject previews: right `[.615,.43,.695,.78]` (foot .655,.78);
  left `[.416,.47,.485,.82]` (foot .4505,.82).
- Original crop `[0,0,1,1]`, plus a right-edge trim `[0,.09,.76,.90]`
  declared 4:3. Both preserve the two anchors; the trim removes the yellow
  foreground pole but loses some of the right-hand facade. No winner set.
- Negative placements: `[.49,.335,.565,.76]` overlaps far roof;
  `[.19,.31,.25,.66]` footpoint lacks the labeled ground cue.

On Draft #21, passed: 2 distinct pixel crops, each with 2 provisional
stand-ins; 2 separate unmarked review JPEGs (909 × 637 and 691 × 517);
both negative placements rejected with the expected codes. Both full-frame
overlays were visually inspected (a silhouette, not a detected person).
The cyan shape was visible; the Chinese label rendered as missing glyphs on
the cloud Linux font setup. This does not establish a Windows or phone bug.

## F20-SMOKE-02: no credible standing surface

Public-domain SpaceX photo via [scikit-image v0.24.0](https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/data/rocket.jpg),
Git blob `32bdd33960c60d012fc71b5fd80a7ccc24d3b906`, 640 × 427. Launchpad
scene with no visible person and no credible continuous foreground standing
region. Use an empty anchor list and offer a synthetic candidate at
`[.62,.47,.71,.82]` in the original crop. Passed: no shortlisted frame,
`footpoint_not_on_visible_ground`; no guide was selected.

## Reproduction and next step

Feed the listed annotations and proposals to `framing_shortlist.shortlist`.
Use the same upright RGB JPEG normalization as `field_pilot.prepare_photo`
before `export_clean_crops` (this helper expects normalized JPEG bytes).
Use `render.render_overlay` with `status=no_subject`, `framing=full_body`,
`basis=open_area`, and a diagnostic-only instruction. Do not commit source
or overlay image copies. For quality evaluation, obtain clearly empty
authorized positives, run blind observer and independent same-pair judge
at a verified zero-cost tier, then collect paired real retakes. Keep
model selection and human outcomes separately reported.
