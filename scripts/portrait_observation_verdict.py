"""Adjudicate the opt-in portrait observation receipt. Offline, no image I/O."""
import argparse
import json
from pathlib import Path

from compare_portrait_judges import (
    TARGETS, keys, nonempty, normalized, result, score_image,
)

VERSION = "portrait-observations-v0.4"
REQUIREMENT_STATUSES = ("pass", "fail", "unknown")
GOAL_SCORES = {"met": 2, "partly_met": 1, "not_met": 0, "unknown": None}


def expected_ids(brief):
    keys(brief, ("goal", "required_elements"))
    nonempty(brief["goal"])
    if not isinstance(brief["required_elements"], list):
        raise ValueError("required_elements must be a list")
    ids = set()
    for element in brief["required_elements"]:
        keys(element, ("id", "criterion"))
        nonempty(element["id"])
        nonempty(element["criterion"])
        if element["id"] in ids:
            raise ValueError("duplicate brief requirement ID")
        ids.add(element["id"])
    return ids


def adjudicate_image(image, ids):
    keys(image, ("scores", "requirements", "goal"))
    goal = image["goal"]
    keys(goal, ("status", "evidence"))
    if goal["status"] not in tuple(GOAL_SCORES):
        raise ValueError("invalid goal status")
    nonempty(goal["evidence"])
    requirements = image["requirements"]
    if not isinstance(requirements, list):
        raise ValueError("requirements must be a list")
    seen = set()
    statuses = set()
    for item in requirements:
        keys(item, ("id", "status", "region", "evidence"))
        nonempty(item["id"])
        if item["id"] not in ids or item["id"] in seen:
            raise ValueError("unexpected or duplicate observation ID")
        seen.add(item["id"])
        state = item["status"]
        if state not in REQUIREMENT_STATUSES:
            raise ValueError("invalid requirement status")
        nonempty(item["evidence"])
        if state != "unknown" or item["region"] is not None:
            nonempty(item["region"])
        statuses.add(state)
    if seen != ids:
        raise ValueError("incomplete required-element observations")

    # Reuse the five-axis evidence validator, not the old intent arithmetic.
    total, _ = score_image({"scores": image["scores"],
                            "intent": {"score": None, "requirements": [],
                                       "reason": "checked separately"}}, 0)
    if "fail" in statuses:
        intent = 0
    elif "unknown" in statuses:
        intent = None
    else:
        intent = GOAL_SCORES[goal["status"]]
    return {"beauty_points": None if total is None else total / 4,
            "intent_score": intent}


def adjudicate(brief, response):
    ids = expected_ids(brief)
    if not isinstance(response, dict) or response.get("version") != VERSION:
        raise ValueError("wrong or missing receipt version")
    if response.get("status") == "abstain":
        keys(response, ("version", "status", "reason"))
        nonempty(response["reason"])
        return {"status": "abstain", "winner": dict.fromkeys(TARGETS, "uncertain")}
    keys(response, ("version", "status", "A", "B"))
    if response["status"] != "ok":
        raise ValueError("invalid review status")
    a, b = (adjudicate_image(response[side], ids) for side in ("A", "B"))
    return {"status": "ok", "A": a, "B": b, "winner": {
        "beauty": result(a["beauty_points"], b["beauty_points"], 5),
        "intent": result(a["intent_score"], b["intent_score"]),
    }}


def adjudicate_swap(brief, forward, reverse):
    expected_ids(brief)
    rows = {}
    for order, receipt in (("forward", forward), ("reverse", reverse)):
        try:
            rows[order] = adjudicate(brief, receipt)
        except ValueError as exc:
            rows[order] = {"status": "invalid", "error": str(exc),
                           "winner": dict.fromkeys(TARGETS, "uncertain")}
    winner = {}
    for target in TARGETS:
        a = rows["forward"]["winner"][target]
        b = normalized(rows["reverse"]["winner"][target], True)
        winner[target] = a if a == b else "uncertain"
    return {"version": VERSION, "orders": rows, "stable_source_order_winner": winner}


def no_duplicate_keys(items):
    out = {}
    for k, v in items:
        if k in out:
            raise ValueError(f"duplicate JSON key: {k}")
        out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", type=Path)
    args = parser.parse_args()
    data = json.loads(args.pair.read_text(encoding="utf-8-sig"),
                      object_pairs_hook=no_duplicate_keys)
    keys(data, ("brief", "forward", "reverse"))
    print(json.dumps(adjudicate_swap(data["brief"], data["forward"], data["reverse"]),
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
