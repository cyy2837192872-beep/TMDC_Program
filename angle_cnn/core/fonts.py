"""Matplotlib CJK font setup with safe fallbacks (no hard dependency on Noto)."""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

from matplotlib import rcParams
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

logger = logging.getLogger(__name__)

# WSL Windows fonts first: NotoSansCJK-Regular.ttc often resolves to the *JP*
# subset via matplotlib, which can miss glyphs like U+2082 (subscript 2) in titles.
# Microsoft YaHei / SimSun typically cover full Chinese + Latin punctuation.
_CJK_FONT_FILE_CANDIDATES: tuple[str, ...] = (
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/msyhbd.ttc",
    "/mnt/c/Windows/Fonts/msjhl.ttc",
    "/mnt/c/Windows/Fonts/simsun.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/mnt/c/Windows/Fonts/msyh.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)

# Cached after first successful load (explicit FontProperties fixes suptitle on some backends).
_cjk_fontproperties: Optional[FontProperties] = None


def _first_existing(paths: Iterable[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def cjk_fontproperties(extra_paths: Optional[Iterable[str]] = None) -> Optional[FontProperties]:
    """
    Return a FontProperties for the first available CJK font file.

    Use this for ``fig.suptitle(..., fontproperties=...)`` — ``rcParams`` alone
    is not always respected by ``Figure.suptitle`` (especially with Agg).
    """
    global _cjk_fontproperties

    if _cjk_fontproperties is not None:
        return _cjk_fontproperties

    env_path = os.environ.get("MPL_CJK_FONT", "").strip()
    candidates: list[str] = []
    if env_path:
        candidates.append(env_path)
    if extra_paths:
        candidates.extend(extra_paths)
    candidates.extend(_CJK_FONT_FILE_CANDIDATES)

    path = _first_existing(candidates)
    if path is None:
        logger.warning(
            "No CJK font file found (set MPL_CJK_FONT or install Noto CJK, "
            "or ensure WSL can read /mnt/c/Windows/Fonts). "
            "Chinese labels may render as boxes."
        )
        rcParams["axes.unicode_minus"] = False
        return None

    try:
        fm.fontManager.addfont(path)
        _cjk_fontproperties = FontProperties(fname=path)
        font_name = _cjk_fontproperties.get_name()
        rcParams["font.sans-serif"] = [font_name] + list(rcParams.get("font.sans-serif", []))
        rcParams["axes.unicode_minus"] = False
        rcParams["mathtext.default"] = "regular"
        return _cjk_fontproperties
    except Exception as exc:  # noqa: BLE001 — font stack is best-effort
        logger.warning("Could not load font from %s: %s", path, exc)
        rcParams["axes.unicode_minus"] = False
        return None


def setup_matplotlib_cjk_font(extra_paths: Optional[Iterable[str]] = None) -> Optional[str]:
    """
    Prefer a CJK-capable font if available; otherwise leave Matplotlib defaults.

    Returns the font family name applied to rcParams['font.sans-serif'][0], or None if fallback.
    """
    fp = cjk_fontproperties(extra_paths=extra_paths)
    if fp is None:
        return None
    return fp.get_name()
