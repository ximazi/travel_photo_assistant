"""Offline portrait judge validation and cross-model comparison. No network."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

WEIGHTS = dict(person_scene=25, composition=25, light_color=20,
               pose_story=20, technical=10)
OUTCOMES = {"A", "B", "tie", "uncertain"}
TARGETS = ("beauty", "intent")


def keys(data, expected):
    if not isinstance(data, dict) or data.keys() != set(expected):
        raise ValueError(f"expected keys: {', '.join(expected)}")


def nonempty(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected nonempty text")


def rating(value, maximum):
    if value is not None and (type(value) is not int or not 0 <= value <= maximum):
        raise ValueError(f"expected null or integer in [0,{maximum}]")


def result(left, right, margin=0):
    if left is None or right is None:
        return "uncertain"
    if abs(left - right) <= margin:
        return "tie"
    return "A" if left > right else "B"


def score_image(image, n_requirements):
    keys(image, ("scores", "intent"))
    keys(image["scores"], WEIGHTS)
    total = 0  # Four times the weighted score; avoids floating-point ties.
    for axis, weight in WEIGHTS.items():
        item = image["scores"][axis]
        keys(item, ("score", "region", "evidence"))
        value = item["score"]
        rating(value, 4)
        nonempty(item["evidence"])
        if value is None:
            if item["region"] is not None:
                raise ValueError("unknown dimension requires a null region")
            total = None
        else:
            nonempty(item["region"])
            if total is not None:
                total += value * weight
    intent = image["intent"]
    keys(intent, ("score", "requirements", "reason"))
    rating(intent["score"], 2)
    nonempty(intent["reason"])
    requirements = intent["requirements"]
    if not isinstance(requirements, list) or len(requirements) != n_requirements:
        raise ValueError("requirement count mismatch")
    statuses = set()
    for requirement in requirements:
        keys(requirement, ("status", "evidence"))
        if requirement["status"] not in ("present", "missing", "unknown"):
            raise ValueError("invalid requirement status")
        statuses.add(requirement["status"])
        nonempty(requirement["evidence"])
    if "missing" in statuses and intent["score"] != 0:
        raise ValueError("visibly missing hard requirement must score 0")
    if "unknown" in statuses and intent["score"] not in (0, None):
        raise ValueError("unknown hard requirement cannot pass")
    return total, intent["score"]


def checked_vote(response, n_requirements):
    if isinstance(response, dict) and response.get("status") == "abstain":
        keys(response, ("status", "reason"))
        nonempty(response["reason"])
        return dict.fromkeys(TARGETS, "uncertain")
    keys(response, ("status", "A", "B", "beauty", "intent"))
    if response["status"] != "ok":
        raise ValueError("invalid status")
    a, b = (score_image(response[x], n_requirements) for x in ("A", "B"))
    expected = {"beauty": result(a[0], b[0], 20), "intent": result(a[1], b[1])}
    for target, winner in expected.items():
        keys(response[target], ("winner", "reason"))
        nonempty(response[target]["reason"])
        if response[target]["winner"] != winner:
            raise ValueError(f"{target} verdict contradicts component scores")
    return expected


def normalized(vote, reverse):
    return {"A": "B", "B": "A"}.get(vote, vote) if reverse else vote


def summarize(pairs, names):
    if not pairs or not names or len(set(names)) != len(names):
        raise ValueError("need pairs and distinct pinned model identifiers")
    counters = defaultdict(Counter)
    votes = {}
    seen = set()
    for row in pairs:
        pair, scene = row["pair_id"], row["scene"]
        nonempty(pair)
        nonempty(scene)
        if pair in seen or scene == "all":
            raise ValueError("duplicate pair_id or reserved scene label")
        seen.add(pair)
        human, models, reqs = row["human"], row["models"], row["requirement_count"]
        keys(human, TARGETS)
        if any(v not in OUTCOMES for v in human.values()):
            raise ValueError("invalid human verdict")
        if type(reqs) is not int or reqs < 0:
            raise ValueError("invalid requirement count")
        if set(models) - set(names):
            raise ValueError("unpinned model in result")
        votes[pair] = {}
        for name in names:
            runs = models.get(name) or {}
            parsed = []
            for order in ("forward", "reverse"):
                try:
                    parsed.append(checked_vote(runs.get(order), reqs))
                except (ValueError, TypeError):
                    parsed.append(None)
            stable = {}
            for target in TARGETS:
                first, other = (entry[target] if entry else None for entry in parsed)
                second = normalized(other, True)
                answered = first is not None and second is not None
                same = answered and first == second
                outcome = first if same else "uncertain"
                stable[target] = outcome
                for group in ("all", scene):
                    c = counters[(name, group, target)]
                    c["pairs"] += 1
                    c["invalid_or_missing_orders"] += sum(entry is None for entry in parsed)
                    c["both_valid"] += int(answered)
                    c["swap_stable"] += int(same)
                    c["decisive_swap_stable"] += int(same and outcome != "uncertain")
                    c["abstained_or_unstable"] += int(outcome == "uncertain")
                    c["human_decided"] += int(human[target] != "uncertain")
                    c["human_decided_model_covered"] += int(human[target] != "uncertain" and outcome != "uncertain")
                    c["matches_human"] += int(human[target] != "uncertain" and human[target] == outcome)
                    c["human_ties"] += int(human[target] == "tie")
                    c["human_tie_model_picked"] += int(human[target] == "tie" and outcome in {"A", "B"})
            votes[pair][name] = stable
    return {"metrics": [{"model": model, "scene": scene, "target": target, **counts}
                        for (model, scene, target), counts in sorted(counters.items())],
            "stable_votes_by_pair": votes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    data = json.loads(args.batch.read_text(encoding="utf-8-sig"))
    print(json.dumps(summarize(data["pairs"], data["model_ids"]),
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
