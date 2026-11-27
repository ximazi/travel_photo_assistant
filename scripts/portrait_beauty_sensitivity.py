"""Describe how volatile an experimental v0.4 beauty verdict is, offline."""
import argparse
import json
from pathlib import Path

from compare_portrait_judges import WEIGHTS, keys, result
from portrait_observation_verdict import adjudicate_swap, no_duplicate_keys


def _source_grade_vectors(raw, reverse):
    return {source: {k: raw[slot]["scores"][k]["score"] for k in WEIGHTS}
            for source, slot in zip("AB", "BA" if reverse else "AB")}


def _diagnose(grades):
    if any(v is None for g in grades.values() for v in g.values()):
        return {"signed_gap_A_minus_B": None, "verdict": "uncertain",
                "single_grade_changes_verdict": None}
    gap = sum((grades["A"][key] - grades["B"][key]) * w for key, w in WEIGHTS.items())
    original = result(gap, 0, 20)
    changes = []
    for side, values in grades.items():
        for axis, grade in values.items():
            for step in (-1, 1):
                if 0 <= grade + step <= 4:
                    changed = gap + (1 if side == "A" else -1) * step * WEIGHTS[axis]
                    after = result(changed, 0, 20)
                    if after != original:
                        changes.append({"source": side, "axis": axis,
                                        "delta_grade": step, "verdict": after})
    return {"signed_gap_A_minus_B": gap / 4, "verdict": original,
            "single_grade_changes_verdict": changes}


def diagnostics(brief, forward, reverse):
    audited = adjudicate_swap(brief, forward, reverse)
    orders, vectors = {}, {}
    for name, raw in (("forward", forward), ("reverse", reverse)):
        status = audited["orders"][name]["status"]
        orders[name] = {"status": status}
        if status == "ok":
            vectors[name] = _source_grade_vectors(raw, name == "reverse")
            orders[name].update(_diagnose(vectors[name]))
    changes = None
    if len(vectors) == 2:
        f, r = vectors["forward"], vectors["reverse"]
        changes = {side: {axis: [f[side][axis], r[side][axis]]
                          for axis in WEIGHTS if f[side][axis] != r[side][axis]}
                   for side in ("A", "B")}
    return {"orders": orders, "observed_grade_drift": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", type=Path)
    args = parser.parse_args()
    data = json.loads(args.pair.read_text(encoding="utf-8-sig"),
                      object_pairs_hook=no_duplicate_keys)
    keys(data, ("brief", "forward", "reverse"))
    print(json.dumps(diagnostics(data["brief"], data["forward"], data["reverse"]), indent=2))


if __name__ == "__main__":
    main()
