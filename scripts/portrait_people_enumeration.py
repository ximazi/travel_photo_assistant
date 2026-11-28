"""Count audit for an explicit exactly-one requirement; not a portrait scorer."""
from compare_portrait_judges import keys, nonempty

VERSION = "people-enumeration-v0.1"


def _box(box):
    if (not isinstance(box, list) or len(box) != 4
            or any(type(n) is not int or not 0 <= n <= 1000 for n in box)
            or box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError("invalid instance box")
    return box


def _overlaps(a, b):
    # Containment also indicates a possible duplicate, not two proven people.
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return False
    small = min((x[2]-x[0])*(x[3]-x[1]) for x in (a, b))
    return 2 * width * height >= small


def count_requirement(image):
    keys(image, ("scan", "scan_evidence", "instances"))
    scan = image["scan"]
    if scan not in ("complete", "incomplete"):
        raise ValueError("invalid scan status")
    nonempty(image["scan_evidence"])
    if not isinstance(image["instances"], list):
        raise ValueError("instances must be a list")
    confirmed, possible = [], []
    for item in image["instances"]:
        keys(item, ("state", "box", "evidence"))
        state = item["state"]
        if state not in ("confirmed", "possible"):
            raise ValueError("invalid instance state")
        nonempty(item["evidence"])
        (confirmed if state == "confirmed" else possible).append(_box(item["box"]))
    distinct_pair = any(not _overlaps(a, b) for i, a in enumerate(confirmed)
                        for b in confirmed[:i])
    if distinct_pair:
        return "fail"
    if scan == "incomplete" or possible or len(confirmed) > 1:
        return "unknown"
    return "pass" if confirmed else "fail"


def audit_pair(receipt):
    keys(receipt, ("version", "A", "B"))
    if receipt["version"] != VERSION:
        raise ValueError("wrong enumeration version")
    return {k: count_requirement(receipt[k]) for k in ("A", "B")}
