"""Optional JSON transport schema for the v0.4 observation experiment."""
from compare_portrait_judges import WEIGHTS
from portrait_observation_verdict import (
    GOAL_SCORES, REQUIREMENT_STATUSES, VERSION, expected_ids,
)


def _object(**fields):
    return {"type": "object", "properties": fields, "required": list(fields),
            "additionalProperties": False}


def _enum(values):
    return {"enum": list(values)}


def observation_schema(brief):
    """Constrain output shape; the adjudicator still validates all semantics."""
    ids = sorted(expected_ids(brief))
    words = {"type": "string", "minLength": 1}
    missing = {"type": "null"}
    version = _enum((VERSION,))
    axis = {"anyOf": [
        _object(score={"type": "integer", "minimum": 0, "maximum": 4},
                region=words, evidence=words),
        _object(score=missing, region=missing, evidence=words),
    ]}
    observations = {"type": "array", "minItems": len(ids), "maxItems": len(ids)}
    if ids:
        observations["items"] = {"anyOf": [
            _object(id=_enum(ids), status=_enum(REQUIREMENT_STATUSES[:2]),
                    region=words, evidence=words),
            _object(id=_enum(ids), status=_enum(("unknown",)),
                    region={"anyOf": [words, missing]}, evidence=words),
        ]}
    image = _object(scores=_object(**dict.fromkeys(WEIGHTS, axis)),
                    requirements=observations,
                    goal=_object(status=_enum(GOAL_SCORES), evidence=words))
    return {"anyOf": [_object(version=version, status=_enum(("ok",)), A=image, B=image),
                      _object(version=version, status=_enum(("abstain",)), reason=words)]}
