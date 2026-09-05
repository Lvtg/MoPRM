from __future__ import annotations

import math
import re
from fractions import Fraction


_CHOICE_RE = re.compile(
    r"(?:final\s+answer|answer|therefore|thus|so)\s*(?:is|:)?\s*[\(\[]?\s*([A-E])\s*[\)\].]?",
    flags=re.IGNORECASE,
)
_ANY_CHOICE_RE = re.compile(r"(?<![A-Z])([A-E])(?![A-Z])", flags=re.IGNORECASE)
_ANSWER_LINE_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)?\s*(.+)$",
    flags=re.IGNORECASE,
)
_LATEX_TEXT_RE = re.compile(r"\\text\{([^{}]*)\}")
_LATEX_FRAC_RE = re.compile(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\{([^{}]+)\}")
_UNIT_SUFFIXES = (
    "centimeters",
    "centimetres",
    "kilograms",
    "millimeters",
    "millimetres",
    "seconds",
    "minutes",
    "degrees",
    "gallons",
    "dollars",
    "meters",
    "metres",
    "inches",
    "pounds",
    "grams",
    "hours",
    "miles",
    "yards",
    "units",
    "feet",
    "foot",
    "inch",
    "mile",
    "yard",
    "days",
    "kgs",
    "lbs",
    "day",
    "sec",
    "min",
    "hrs",
    "cm",
    "mm",
    "kg",
    "lb",
    "ft",
    "m",
    "s",
)


def strip_boxed(text: str) -> str:
    marker = r"\boxed{"
    start = text.find(marker)
    if start < 0:
        return text
    idx = start + len(marker)
    depth = 1
    chars: list[str] = []
    while idx < len(text) and depth > 0:
        char = text[idx]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth > 0:
                chars.append(char)
        else:
            chars.append(char)
        idx += 1
    return "".join(chars).strip() if chars else text


def extract_choice(text: str) -> str | None:
    match = _CHOICE_RE.search(text)
    if match:
        return match.group(1).upper()
    matches = _ANY_CHOICE_RE.findall(text)
    if matches:
        return matches[-1].upper()
    return None


def extract_final_answer(text: str) -> str:
    boxed = strip_boxed(text)
    if boxed != text:
        return boxed

    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        match = _ANSWER_LINE_RE.search(stripped)
        if match:
            return match.group(1).strip()

    boxed_match = re.search(r"\\boxed\{(.+?)\}", text, flags=re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).strip()

    return text.strip()


def has_explicit_final_answer(text: str) -> bool:
    boxed = strip_boxed(text)
    if boxed != text:
        return True
    if re.search(r"\\boxed\{(.+?)\}", text, flags=re.IGNORECASE):
        return True
    return any(_ANSWER_LINE_RE.search(line.strip()) for line in text.splitlines())


def normalize_latex(answer: str) -> str:
    value = str(answer)
    value = strip_boxed(value)
    value = _LATEX_TEXT_RE.sub(r"\1", value)

    previous = None
    while previous != value:
        previous = value
        value = _LATEX_FRAC_RE.sub(r"\1/\2", value)
        value = _LATEX_SQRT_RE.sub(r"sqrt(\1)", value)

    replacements = {
        "\\(": "",
        "\\)": "",
        "\\[": "",
        "\\]": "",
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\!": "",
        "\\;": "",
        "\\:": "",
        "\\cdot": "*",
        "\\times": "*",
        "\\pi": "pi",
        "^\\circ": "",
        "^{\\circ}": "",
        "\\circ": "",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def normalize_answer(answer: str) -> str:
    value = normalize_latex(str(answer))
    value = value.strip().lower()
    value = value.replace("$", "")
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
    value = value.strip()
    value = re.sub(r"^(?:the\s+)?answer\s+(?:is|:)\s*", "", value)
    value = re.sub(r"^(?:option|choice)\s+", "", value)
    value = value.strip(" .;:\n\t[]{}")
    value = re.sub(r"\s+", "", value)
    return value


def _parse_normalized_numeric(normalized: str) -> float | None:
    if not normalized:
        return None
    normalized = normalized.replace("%", "/100")
    if normalized.startswith("(") and normalized.endswith(")") and normalized.count("(") == 1:
        normalized = normalized[1:-1]
    frac_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", normalized)
    if frac_match:
        numerator = Fraction(frac_match.group(1))
        denominator = Fraction(frac_match.group(2))
        if denominator == 0:
            return None
        return float(numerator / denominator)
    try:
        return float(normalized)
    except ValueError:
        return None


def _strip_numeric_unit_suffix(normalized: str) -> str:
    for suffix in _UNIT_SUFFIXES:
        if not normalized.endswith(suffix):
            continue
        stripped = normalized[: -len(suffix)]
        if stripped and _parse_normalized_numeric(stripped) is not None:
            return stripped
    return normalized


def parse_numeric(answer: str) -> float | None:
    normalized = normalize_answer(answer)
    parsed = _parse_normalized_numeric(normalized)
    if parsed is not None:
        return parsed
    stripped = _strip_numeric_unit_suffix(normalized)
    if stripped != normalized:
        return _parse_normalized_numeric(stripped)
    return None


def check_answer(prediction: str, gold: str, domain: str | None = None, tol: float = 1e-6) -> bool:
    pred_text = extract_final_answer(str(prediction))
    gold_text = extract_final_answer(str(gold))

    gold_choice = extract_choice(gold_text)
    pred_choice = extract_choice(pred_text)
    if domain == "logic" or gold_choice is not None:
        return pred_choice is not None and pred_choice == gold_choice

    pred_num = parse_numeric(pred_text)
    gold_num = parse_numeric(gold_text)
    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=tol, abs_tol=tol)

    return normalize_answer(pred_text) == normalize_answer(gold_text)
