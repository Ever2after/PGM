"""Final-answer extraction, equivalence, and online answer groups."""
from __future__ import annotations

from functools import lru_cache

import re


def boxed_span(text):
    """Return body and character spans of the last COMPLETE boxed/fbox answer."""
    found = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        start = match.end()
        depth = 1
        i = start
        while i < len(text):
            # An escaped brace is mathematical content, not a grouping delimiter.
            preceding = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                preceding += 1
                j -= 1
            if preceding % 2 == 0:
                depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                found.append({"answer": text[start:i].strip(), "box_start": match.start(), "body_start": start, "body_end": i})
                break
            i += 1
    # The original MATH train set also contains valid final forms like $\boxed 2$.
    for match in re.finditer(r"\\(?:boxed|fbox)\s+([A-Za-z0-9+\-][^$\n]*)", text):
        answer = match.group(1).strip()
        found.append({"answer": answer, "box_start": match.start(), "body_start": match.start(1), "body_end": match.start(1) + len(match.group(1).rstrip())})
    return max(found, key=lambda x: x["box_start"]) if found else None


def normalize_answer(answer):
    if answer is None:
        return None
    answer = answer.strip().strip("$")
    answer = answer.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    answer = answer.replace(r"\left", "").replace(r"\right", "")
    return re.sub(r"\s+", "", answer)


@lru_cache(maxsize=32768)
def parsed_answer(answer):
    from math_verify import LatexExtractionConfig, parse
    return parse(r"\boxed{" + answer + "}", extraction_config=[LatexExtractionConfig()], parsing_timeout=2)


@lru_cache(maxsize=65536)
def equivalent(a, b):
    """Compare generated answers without access to gold labels."""
    if a is None or b is None or not a or not b:
        return False
    if normalize_answer(a) == normalize_answer(b):
        return True
    from math_verify import verify
    try:
        pa, pb = parsed_answer(a), parsed_answer(b)
        # verify can be directional for non-scalar objects; grouping is symmetric.
        return bool(pa and pb and verify(pa, pb, timeout_seconds=2) and verify(pb, pa, timeout_seconds=2))
    except Exception:
        return False


def answer_groups(answers):
    """Online equivalence classes, with first-occurrence tie breaking; no gold."""
    representatives, assignments = [], []
    for answer in answers:
        if not answer:
            assignments.append(-1)
            continue
        match = next((k for k, prior in enumerate(representatives) if equivalent(prior, answer)), None)
        if match is None:
            match = len(representatives)
            representatives.append(answer)
        assignments.append(match)
    return representatives, assignments
