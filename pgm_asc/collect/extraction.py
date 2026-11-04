"""Answer extraction and ReASC self-certainty, as in the official ReASC notebooks."""
from __future__ import annotations

import math
import re

import numpy as np


def first_box(text):
    start = text.find(r"\boxed{")
    if start < 0:
        return None
    i = start + len(r"\boxed{")
    depth = 1
    body = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        body.append(c)
        i += 1
    return "".join(body)


def clean(text, dataset="math"):
    if dataset == "gsm8k":
        numbers = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
        return numbers[-1] if numbers else None
    value = first_box(text)
    if value is None:
        return ""
    return value.replace(" ", "")


def prediction(response, extracted, dataset="math"):
    """Final answer string: boxed answer in the response, else in the extraction output, else the raw extraction."""
    if dataset == "gsm8k":
        return clean(extracted.replace(" ", "").strip(), dataset)
    value = clean(response, dataset)
    if value == "":
        value = clean(extracted, dataset)
    if value == "":
        value = extracted.replace(" ", "").strip()
    return value


def self_certainty(top_logprobs):
    """Per-token self-certainty from the top-K log-probabilities: -log K - mean(logprob)."""
    k = len(top_logprobs)
    return -math.log(k) - sum(top_logprobs) / k


def confidence_groups(token_scores, window=128):
    """ReASC confidence: lowest and bottom-10% means of stride-1 windows of 128 tokens."""
    values = np.asarray(token_scores, dtype=np.float64)
    if not len(values):
        return {"lowest": 0., "bottom10": 0., "empty": True}
    c = np.r_[0., np.cumsum(values)]
    start = np.arange(len(values))
    end = np.minimum(start + window, len(values))
    averages = (c[end] - c[start]) / (end - start)
    count = max(1, len(averages) // 10)
    return {"lowest": float(averages.min()), "bottom10": float(np.partition(averages, count - 1)[:count].mean()), "empty": False}


def boxed_span(text):
    """Body and character spans of the last complete \\boxed{...} / \\fbox{...} answer (for teacher forcing)."""
    found = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        start = match.end()
        depth = 1
        i = start
        while i < len(text):
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
    for match in re.finditer(r"\\(?:boxed|fbox)\s+([A-Za-z0-9+\-][^$\n]*)", text):
        answer = match.group(1).strip()
        found.append({"answer": answer, "box_start": match.start(), "body_start": match.start(1),
                      "body_end": match.start(1) + len(match.group(1).rstrip())})
    return max(found, key=lambda x: x["box_start"]) if found else None
