from typing import FrozenSet, Set
import os
from pathlib import Path

from .charsets_data import (
    GB2312_CODEPOINTS,
    BIG5_CODEPOINTS,
    JISX0208_CODEPOINTS,
    KSX1001_CODEPOINTS,
)


SUPPORTED_CHARSETS = frozenset([
    "gb2312",
    "gbk",
    "big5",
    "jisx0208",
    "ksx1001",
    "target",
])


def _load_charset_from_file(file_path: str) -> FrozenSet[int]:
    codepoints: Set[int] = set()
    charset_path = Path(file_path)

    if not charset_path.exists():
        raise FileNotFoundError(f"Charset file not found: {file_path}")

    with open(charset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('U+'):
                codepoints.add(int(line[2:], 16))
            elif line.startswith('0x'):
                codepoints.add(int(line, 16))
            elif len(line) == 1:
                codepoints.add(ord(line))

    return frozenset(codepoints)


def get_charset_codepoints(charset_name: str) -> FrozenSet[int]:
    if not charset_name:
        return frozenset()

    charset_path = Path(charset_name)
    if charset_path.exists() and charset_path.is_file():
        return _load_charset_from_file(charset_name)

    charset_lower = charset_name.lower()

    if charset_lower not in SUPPORTED_CHARSETS:
        raise ValueError(
            f"Unknown charset: {charset_name}. "
            f"Available: {', '.join(sorted(SUPPORTED_CHARSETS))}, or a file path"
        )

    if charset_lower == "target":
        target_charset_path = Path("data/base/target_charset.txt")
        if target_charset_path.exists():
            return _load_charset_from_file(str(target_charset_path))
        raise FileNotFoundError(f"Target charset file not found: {target_charset_path}")
    if charset_lower == "gb2312":
        return GB2312_CODEPOINTS
    if charset_lower == "gbk":
        return GB2312_CODEPOINTS
    if charset_lower == "big5":
        return BIG5_CODEPOINTS
    if charset_lower == "jisx0208":
        return JISX0208_CODEPOINTS
    if charset_lower == "ksx1001":
        return KSX1001_CODEPOINTS

    raise ValueError(f"Unknown charset: {charset_name}")
