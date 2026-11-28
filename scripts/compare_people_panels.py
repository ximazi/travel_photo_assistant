"""Offline scorecard for two-order, matched people-enumeration experiments."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from compare_portrait_judges import keys, nonempty
from portrait_observation_verdict import no_duplicate_keys
from portrait_people_enumeration import audit_pair

LABELS = ("one", "zero", "multiple", "uncertain")


def _parse(data):
    return json.loads(data, object_pairs_hook=no_duplicate_keys)


def _photo(photo):
    keys(photo, ("sha256", "votes"))
    digest, votes = photo["sha256"], photo["votes"]
    if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("bad photo SHA-256")
    if not isinstance(votes, list) or len(votes) != 2 or any(v not in LABELS for v in votes):
        raise ValueError("two independently supplied labels are required")
    status = votes[0] if votes[0] == votes[1] else "disagreement"
    return digest, status


def _arm(arm, expected, reverse=False):
    if arm is None:
        return ("invalid", "invalid")
    keys(arm, ("image_sha256", "receipt"))
    if arm["image_sha256"] != list(expected):
        raise ValueError("run image bytes or attachment order differ from locked panel")
    try:
        verdicts = audit_pair(arm["receipt"])
    except ValueError:
        return ("invalid", "invalid")
    order = ("B", "A") if reverse else ("A", "B")
    return tuple(verdicts[k] for k in order)


def _ratio(num, den):
    return {"n": num, "d": den}


def scorecard(panel_bytes, submissions):
    panel = _parse(panel_bytes)
    digest = hashlib.sha256(panel_bytes).hexdigest()
    keys(panel, ("version", "pairs"))
    if panel["version"] != "people-panel-v0.1":
        raise ValueError("wrong panel version")
    if not isinstance(panel["pairs"], dict) or not panel["pairs"]:
        raise ValueError("a nonempty locked panel is required")
    locked = {}
    seen = set()
    for pair_id, pair in panel["pairs"].items():
        nonempty(pair_id)
        keys(pair, ("A", "B"))
        a, b = (_photo(pair[k]) for k in ("A", "B"))
        if a[0] == b[0] or a[0] in seen or b[0] in seen:
            raise ValueError("repeated photo in panel")
        seen.update((a[0], b[0]))
        locked[pair_id] = (a, b)

    if not isinstance(submissions, dict) or not submissions:
        raise ValueError("model submissions are required")
    labels = Counter(label for pair in locked.values() for _, label in pair)
    n = sum(labels.values())
    report = {"panel_sha256": digest, "human_labels": dict(labels), "models": {}}
    for name, run in submissions.items():
        nonempty(name)
        keys(run, ("panel_sha256", "pairs"))
        if run["panel_sha256"] != digest:
            raise ValueError("submission refers to a different locked panel")
        keys(run["pairs"], locked)
        tallies, rows = Counter(), {}
        for key, pair in locked.items():
            arms = run["pairs"][key]
            keys(arms, ("forward", "reverse"))
            (a, ha), (b, hb) = pair
            forward = _arm(arms["forward"], (a, b))
            reverse = _arm(arms["reverse"], (b, a), reverse=True)
            tallies["valid_orders"] += (forward[0] != "invalid") + (reverse[0] != "invalid")
            rows[key] = {}
            for side, human, f, r in zip("AB", (ha, hb), forward, reverse):
                if "invalid" in (f, r):
                    state = "invalid"
                elif f == r:
                    state = f
                else:
                    state = "order_disagreement"
                rows[key][side] = {"human": human, "forward": f, "reverse": r, "stable": state}
                tallies[state] += 1
                if human in ("zero", "multiple") and "pass" in (f, r):
                    tallies["false_pass_either_order"] += 1
                if human in ("zero", "multiple") and state == "pass":
                    tallies["false_stable_pass"] += 1
                if human == "one" and state == "pass":
                    tallies["correct_stable_pass"] += 1
                if human == "one" and state == "fail":
                    tallies["false_stable_fail"] += 1
        neg = labels["zero"] + labels["multiple"]
        report["models"][name] = {
            "valid_orders": _ratio(tallies["valid_orders"], 2 * len(locked)),
            "negative_false_pass_either_order": _ratio(tallies["false_pass_either_order"], neg),
            "negative_false_stable_pass": _ratio(tallies["false_stable_pass"], neg),
            "one_correct_stable_pass": _ratio(tallies["correct_stable_pass"], labels["one"]),
            "one_false_stable_fail": _ratio(tallies["false_stable_fail"], labels["one"]),
            "stable_unknown": _ratio(tallies["unknown"], n),
            "order_disagreement": _ratio(tallies["order_disagreement"], n),
            "invalid_photo_pairs": _ratio(tallies["invalid"], n),
            "rows": rows,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("submissions", type=Path)
    args = parser.parse_args()
    print(json.dumps(scorecard(args.panel.read_bytes(), _parse(args.submissions.read_bytes())),
                     indent=2))


if __name__ == "__main__":
    main()
