"""Check that the model's JSON answer has the right shape before the app uses it."""
import json
import re

CONFIDENCE_LEVELS = ("high", "medium", "low")
ALLOWED_KEYS = {"answer", "found", "citations", "confidence"}


class ValidationFailure(ValueError):
    """The model reply is not valid JSON or does not follow the answer schema."""


def parse_json(text):
    """Parse the reply as JSON (code fences around it are tolerated)."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValidationFailure(f"Reply is not valid JSON ({error.msg}).")


def validate_answer(data, allowed_pages):
    """Return a clean answer dict, or raise ValidationFailure listing every problem found."""
    if not isinstance(data, dict):
        raise ValidationFailure("The top-level JSON value must be an object.")
    errors = []
    extra = set(data) - ALLOWED_KEYS
    if extra:
        errors.append(f"unexpected keys {sorted(extra)}")

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        errors.append("'answer' must be a non-empty string")

    found = data.get("found")
    if not isinstance(found, bool):
        errors.append("'found' must be true or false")

    citations = data.get("citations")
    if not isinstance(citations, list) or not all(isinstance(c, int) and not isinstance(c, bool) for c in citations):
        errors.append("'citations' must be a list of page numbers (integers)")
        citations = []
    else:
        outside = [c for c in citations if c not in allowed_pages]
        if outside:
            errors.append(f"'citations' contains pages {outside} that are not in the context "
                          f"(allowed pages: {sorted(allowed_pages)})")
    if found is True and not citations:
        errors.append("'citations' must list at least one page when found is true")
    if found is False and citations:
        errors.append("'citations' must be empty when found is false")

    confidence = data.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        errors.append(f"'confidence' must be one of {list(CONFIDENCE_LEVELS)}")

    if errors:
        raise ValidationFailure("; ".join(errors))
    return {"answer": answer.strip(), "found": found,
            "citations": sorted(set(citations)), "confidence": confidence}
