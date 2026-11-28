"""Offline blind stage-one aesthetic coverage. No model calls or auto-promotion."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

AXES = ("composition", "light_color", "pose_story")
FIELDS = ("reviewer_id", "image_id", "axis", "grade", "region", "pixel_evidence", "extra_effect")
GRADES = ("", "NA", "0", "1", "2", "3", "4")
BUCKETS = ("consensus_3", "consensus_4", "agreed_0_to_2", "boundary_split",
           "other_disagreement", "unavailable", "insufficient_4_evidence", "insufficient_evidence")


def category(a, b):
    """Structural evidence checks do not verify whether observations are true."""
    if a is None or b is None or any(x["grade"] in ("", "NA") for x in (a, b)):
        return "unavailable"
    ga, gb = a["grade"], b["grade"]
    if ga != gb:
        return "boundary_split" if {ga, gb} == {"3", "4"} else "other_disagreement"
    fields = ("region", "pixel_evidence", "extra_effect") if ga == "4" else ("region", "pixel_evidence")
    if any(not r[f].strip() for r in (a, b) for f in fields):
        return "insufficient_4_evidence" if ga == "4" else "insufficient_evidence"
    return {"3": "consensus_3", "4": "consensus_4"}.get(ga, "agreed_0_to_2")


def load_ballot(path, photo_ids):
    ballot = {}
    reviewers = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("wrong stage-one CSV columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("malformed stage-one row")
            row = {k: v.strip() for k, v in row.items()}
            key = (row["image_id"], row["axis"])
            if key[0] not in photo_ids or key[1] not in AXES or key in ballot:
                raise ValueError("unknown or duplicate photo-axis row")
            if row["grade"] not in GRADES:
                raise ValueError("invalid or fractional grade")
            if not row["reviewer_id"] and row["grade"]:
                raise ValueError("submitted grade without reviewer ID")
            if row["reviewer_id"]:
                reviewers.add(row["reviewer_id"].casefold())
            ballot[key] = row
    if len(reviewers) > 1:
        raise ValueError("one reviewer only per CSV")
    return next(iter(reviewers), None), ballot


def tally(photo_ids, paths):
    if not photo_ids:
        raise ValueError("empty photo manifest")
    (name_a, a), (name_b, b) = [load_ballot(p, photo_ids) for p in paths]
    if name_a is not None and name_a == name_b:
        raise ValueError("same reviewer in both ballots")
    out = {}
    for axis in AXES:
        counts = Counter(category(a.get((p, axis)), b.get((p, axis))) for p in photo_ids)
        out[axis] = {"eligible_photo_rows": len(photo_ids),
                     **{k: counts[k] for k in BUCKETS}}
    return {"ballot_reviewers_present": int(name_a is not None) + int(name_b is not None),
            "axes": out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="private operator JSON containing samples keyed by ID")
    parser.add_argument("ballots", nargs=2, type=Path, help="two separately returned stage-one CSV files")
    args = parser.parse_args()
    samples = json.loads(args.manifest.read_text(encoding="utf-8-sig"))["samples"]
    if not isinstance(samples, dict):
        raise ValueError("invalid locked sample inventory")
    print(json.dumps(tally(set(samples), args.ballots), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
