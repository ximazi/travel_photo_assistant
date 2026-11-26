# Empty-frame portrait placement and visual framing review

Refs #1, #8, #10. This is a field-pilot feature, not a mobile release.

- `field_pilot --compare-empty-scene` holds the same valid observation for
  prompt-only, fixed-card and retrieved-card proposals. A scene with actual but
  unselected people is not an empty scene and does not proceed to the other arms.
  Each accepted arm renders a provisional person silhouette, not a detection.
  The guide's full-body foot midpoint and enclosing visible possible-ground
  anchors are diagnostic fields only. Neither an anchor nor a silhouette proves
  safe access, successful movement, or a new frame. Keep on-site permission
  separate from this preview. The arms are not guaranteed to propose distinct
  compositions. No winner is auto-applied.
- In an empty scene compare the unchanged view against the strongest plausible
  alternative. Set aside cases with no credible standing surface. For a crop
  candidate, hold all other aspects fixed and use only real source pixels.
  Retain the original as a baseline. Before any future automatic candidate
  shortlist, reject candidates cutting the proposed subject or required
  landmark, with pixel aspect mismatch or excessive resolution loss. A changed
  camera position requires real new photos; a crop or synthesized preview
  cannot validate it.
- `python -m travel_photo_assistant.framing_judge --a A.jpg --b B.jpg --intent
  "environmental travel portrait" --model MODEL --out PRIVATE.json
  --confirm-free-project --allow-both-images-to-gemini` runs a separate
  visual judge with reversed A/B ordering, maximum two requests. Use clean
  photographs or real-pixel crops with no arrows, IDs, overlays, plan text,
  card details, or ground-truth labels. Present the shared evaluation intent
  only. Do not mix photographs with silhouettes in the same scored pair.
  The judge allows ties, neither, and uncertainty. If the preference changes
  with order, abstain. A consistent winner is still only a model preference.
  This command is opt-in and needs prior authorization for BOTH images and
  an operator-verified no-billing Gemini project. No other vendor is wired.
- Trial: start with empty tree-lined path, open plaza/architecture, coast and
  night scenes; include unsafe/blocked ground, people with no selected subject,
  and a clipped-landmark negative. Record silhouette coverage, foot-ground
  alignment versus annotated acceptable regions, false person detections,
  landmark retention, no-op wins and invalid placements. Then capture
  same-location unguided vs guided real retakes; blind ≥2 human raters to
  condition, record win/tie/loss and unsafe proposals. Report model agreement,
  abstention, order disagreement and human improvement separately, with sample
  sizes. Use separate locations for tuning and holdout. JEV-like models can be
  screened on those same labeled pairs, not assumed to generalize from generic
  image quality or attribute scores.

