"""Clean OCR output and validate it as an Indian vehicle registration number.

Indian formats handled:
* standard:  SS DD L(0-3) NNNN      e.g. MH 12 AB 1234, DL 3C AB 1234 -> MH12AB1234, DL3CAB1234
* BH series: YY BH NNNN L(1-2)      e.g. 22 BH 1234 AA

OCR often confuses look-alike characters (O/0, I/1, S/5, B/8, Z/2, G/6). Because each position of a
plate is known to be a letter or a digit, we try every split of the string into the format's parts
and repair characters in the wrong class. The candidate with the fewest repairs and a real state code
wins. If nothing fits, the cleaned raw text is returned with ``valid=False`` (still useful to a human).
"""
from __future__ import annotations

import re

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA",
    "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR",
    "TS", "UK", "UA", "UP", "WB",
}
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "J": "1", "T": "1", "Z": "2",
            "S": "5", "B": "8", "G": "6", "A": "4"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A", "7": "T"}


def clean(raw: str) -> str:
    """Upper-case, drop everything except A-Z/0-9, and drop the 'IND' mark printed on HSRP plates."""
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if s.startswith("IND") and len(s) > 9:
        s = s[3:]
    return s


def _coerce(part: str, kind: str) -> tuple[str, int] | None:
    """Force ``part`` to letters ('L') or digits ('D'); return (fixed, number_of_repairs) or None."""
    out, fixes = [], 0
    for ch in part:
        if kind == "D":
            if ch.isdigit():
                out.append(ch)
            elif ch in TO_DIGIT:
                out.append(TO_DIGIT[ch]); fixes += 1
            else:
                return None
        else:
            if ch.isalpha():
                out.append(ch)
            elif ch in TO_LETTER:
                out.append(TO_LETTER[ch]); fixes += 1
            else:
                return None
    return "".join(out), fixes


def _candidates(s: str):
    n = len(s)
    # standard: state(2L) district(1-2D, last char may be a letter as in DL3C) series(0-3L) number(4D)
    for d in (1, 2):
        for series in range(0, 4):
            if 2 + d + series + 4 != n:
                continue
            parts = [(s[0:2], "L"), (s[2:2 + d], "D"), (s[2 + d:2 + d + series], "L"), (s[n - 4:], "D")]
            # almost every district code has 2 digits: a 1-digit split must beat it by more than one
            # repair (else "TS0TEA2133" would win over "TS01EA2133")
            yield parts, "standard", 1.5 if d == 1 else 0.0
            if d == 2:   # 'DL 3C' style (Delhi only): district digit followed by a letter
                yield [(s[0:2], "L"), (s[2:3], "D"), (s[3:4 + series], "L"), (s[n - 4:], "D")], "delhi", 0.0
    # Bharat series: YY BH NNNN LL
    for tail in (1, 2):
        if 2 + 2 + 4 + tail == n:
            yield [(s[0:2], "D"), (s[2:4], "L"), (s[4:8], "D"), (s[8:], "L")], "bh", 0.0


def normalise(raw: str) -> tuple[str, bool]:
    """Return (plate text, is_valid_indian_format)."""
    s = clean(raw)
    best = _best(s)
    if best is not None:
        return best[0], True
    # the 'IND' mark can be read between the two plate lines (GJ03 IND ED4373): drop it, then repair as usual
    if "IND" in s[2:]:
        best = _best(s[:2] + s[2:].replace("IND", ""))
        if best is not None:
            return best[0], True
    # OCR often reads text around the plate too (the 'IND' mark, dealer name / phone number below it):
    # look for a valid number inside the string. Only exact matches (no repaired characters), so random
    # text does not turn into a plate; longer matches first.
    for t in dict.fromkeys((s.replace("IND", ""), s)):
        for n in range(min(11, len(t)), 7, -1):
            for i in range(len(t) - n + 1):
                b = _best(t[i:i + n])
                if b is not None and b[1] == 0:
                    return b[0], True
    return s, False


def _best(s: str) -> tuple[str, int] | None:
    """Best valid reading of the whole string ``s``: (text, number_of_repairs) or None."""
    best = None
    for parts, kind, penalty in _candidates(s):
        fixed, fixes = [], 0
        for part, k in parts:
            r = _coerce(part, k)
            if r is None:
                break
            fixed.append(r[0]); fixes += r[1]
        else:
            text = "".join(fixed)
            if kind == "standard" and text[:2] not in STATE_CODES:
                continue
            if kind == "delhi" and text[:2] != "DL":
                continue
            if kind == "bh" and text[2:4] != "BH":
                continue
            if best is None or fixes + penalty < best[2]:
                best = (text, fixes, fixes + penalty)
    return best[:2] if best else None
