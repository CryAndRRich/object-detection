"""Mapping between category names and their token positions in a caption.

An OD-style caption is built by the dataset as ``" . ".join(categories) + " ."``.
Both the matcher and the post-processor need, for each category, a 0/1 vector over
the 256 text-token slots saying which tokens belong to it.

**Why the spans are computed and not searched.** The obvious implementation is
``caption.find(category)``, which is what upstream does -- and it is wrong
whenever one category name is a substring of another. On COCO, ``"car"`` is a
prefix of ``"carrot"``: with the prompt ``"carrot . car ."``, ``find("car")``
returns 0, so every ``car`` box gets supervised against the ``carrot`` tokens.
The same trap exists for ``"bus"``/``"business"``-style vocabularies and for
``"person"`` vs ``"person on horse"`` in custom label sets.

Because we build the caption ourselves, the character offset of category ``i`` is
known exactly: it is the running sum of the preceding names plus separators. That
is what ``category_char_spans`` returns, and it cannot be fooled by substrings.
"""

import warnings
from typing import List, Sequence, Tuple

import torch
from torch import Tensor

SEPARATOR = " . "
MAX_TEXT_LEN = 256


def build_caption(cat_list: Sequence[str], separator: str = SEPARATOR) -> str:
    """The one place the caption format is defined. Keep dataset and eval in sync."""
    return separator.join(cat_list) + " ."


def category_char_spans(cat_list: Sequence[str], caption: str, separator: str = SEPARATOR) -> List[Tuple[int, int]]:
    """Character span ``[start, end)`` of every category in ``caption``.

    Derived from the construction of the caption, then verified against it. If a
    caption was built some other way and the spans do not line up, falls back to a
    forward search from the current cursor (still substring-safe for the common
    case) and warns once per call.
    """
    spans: List[Tuple[int, int]] = []
    cursor = 0
    drifted = False

    for name in cat_list:
        if caption[cursor : cursor + len(name)] == name:
            start = cursor
        else:
            found = caption.find(name, cursor)
            if found < 0:
                found = caption.find(name)
            if found < 0:
                spans.append((-1, -1))
                drifted = True
                continue
            start = found
            drifted = True
        spans.append((start, start + len(name)))
        cursor = start + len(name) + len(separator)

    if drifted:
        warnings.warn(
            "caption does not match the expected ' . '-joined layout; category token spans "
            "were recovered by search and may be wrong for substring category names",
            stacklevel=2,
        )
    return spans


def _token_span(tokenized, start: int, end: int) -> Tuple[int, int]:
    """Map a character span to an inclusive token span, tolerating punctuation.

    ``char_to_token`` returns ``None`` for characters that fall inside a stripped
    region (whitespace, some punctuation), so the end is walked backwards a couple
    of characters before giving up.
    """
    if start < 0:
        return -1, -1

    beg_pos = tokenized.char_to_token(start)
    end_pos = None
    for offset in (1, 2, 3):
        end_pos = tokenized.char_to_token(end - offset)
        if end_pos is not None:
            break
    if beg_pos is None or end_pos is None or beg_pos < 0 or end_pos < beg_pos:
        return -1, -1
    return beg_pos, end_pos


def create_positive_map(
    tokenized,
    label_ids: Sequence[int],
    cat_list: Sequence[str],
    caption: str,
    max_text_len: int = MAX_TEXT_LEN,
) -> Tensor:
    """One row per entry of ``label_ids``, marking that category's tokens.

    Args:
        tokenized: tokenizer output for ``caption`` (needs ``char_to_token``).
        label_ids: indices into ``cat_list``.
        cat_list: category names, in the order they appear in ``caption``.

    Returns:
        (len(label_ids), max_text_len) float 0/1.
    """
    positive_map = torch.zeros((len(label_ids), max_text_len), dtype=torch.float)
    spans = category_char_spans(cat_list, caption)

    for row, label in enumerate(label_ids):
        label = int(label)
        if label < 0 or label >= len(spans):
            continue
        beg_pos, end_pos = _token_span(tokenized, *spans[label])
        if beg_pos < 0:
            continue
        positive_map[row, beg_pos : min(end_pos + 1, max_text_len)] = 1.0
    return positive_map


def create_positive_map_from_span(tokenized, token_spans, max_text_len: int = MAX_TEXT_LEN) -> Tensor:
    """Positive map from explicit character spans, for phrase-grounding style input.

    ``token_spans`` is a list (one per output row) of lists of ``(start, end)``
    character spans; a row may cover several disjoint spans.
    """
    positive_map = torch.zeros((len(token_spans), max_text_len), dtype=torch.float)
    for row, spans in enumerate(token_spans):
        for start, end in spans:
            beg_pos, end_pos = _token_span(tokenized, start, end)
            if beg_pos < 0:
                continue
            positive_map[row, beg_pos : min(end_pos + 1, max_text_len)] = 1.0
    return positive_map / (positive_map.sum(-1, keepdim=True) + 1e-6)


__all__ = [
    "MAX_TEXT_LEN",
    "SEPARATOR",
    "build_caption",
    "category_char_spans",
    "create_positive_map",
    "create_positive_map_from_span",
]
