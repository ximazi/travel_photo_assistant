"""Opt-in exact-one experiment. This checks evidence, not the actual headcount."""
from portrait_people_enumeration import _box
from compare_portrait_judges import keys, nonempty

VERSION = "people-enumeration-v0.2"


def _inside(inner, outer):
    return (outer[0] <= inner[0] < inner[2] <= outer[2]
            and outer[1] <= inner[1] < inner[3] <= outer[3])


def _separate(a, b):
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def inspect(image):
    keys(image, ("scan", "scan_evidence", "observations"))
    if image["scan"] not in ("complete", "incomplete"):
        raise ValueError("invalid scan status")
    nonempty(image["scan_evidence"])
    rows = image["observations"]
    if not isinstance(rows, list):
        raise ValueError("observations must be a list")
    heads = []
    people = 0
    unresolved = image["scan"] == "incomplete"
    for row in rows:
        keys(row, ("kind", "box", "evidence", "head_cues"))
        kind = row["kind"]
        nonempty(row["evidence"])
        box = _box(row["box"])
        cues = row["head_cues"]
        if not isinstance(cues, list):
            raise ValueError("head_cues must be a list")
        if ((kind == "person" and len(cues) > 1)
                or (kind == "group" and len(cues) > 2)
                or (kind == "uncertain" and cues)
                or kind not in ("person", "group", "uncertain")):
            raise ValueError("invalid observation kind or cue count")
        people += kind == "person"
        unresolved |= kind != "person"
        for cue in cues:
            keys(cue, ("box", "evidence"))
            head = _box(cue["box"])
            nonempty(cue["evidence"])
            if not _inside(head, box):
                raise ValueError("head cue outside its observation")
            heads.append(head)
    # Never add group and person counts: the same person can appear in both.
    pair = any(_separate(a, b) for i, a in enumerate(heads) for b in heads[:i])
    bound = 2 if pair else int(bool(people or heads))
    if pair:
        verdict = "fail"
    elif unresolved or people > 1:
        verdict = "unknown"
    else:
        verdict = "pass" if people else "fail"
    return {"verdict": verdict, "proven_at_least": bound}


def audit_pair(receipt):
    keys(receipt, ("version", "A", "B"))
    if receipt["version"] != VERSION:
        raise ValueError("wrong typed enumeration version")
    return {k: inspect(receipt[k]) for k in ("A", "B")}
