from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ModalityResult:
    score: float
    label: str
    details: Dict[str, Any]
