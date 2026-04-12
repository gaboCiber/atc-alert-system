"""
Configuraciones para transcripción ASR.
"""

from .prompts import (
    DEFAULT_ATC_PROMPT,
    MINIMAL_ATC_PROMPT,
    EXTENDED_ATC_PROMPT,
    get_prompt,
    create_custom_prompt,
    AVAILABLE_PROMPTS,
)

__all__ = [
    "DEFAULT_ATC_PROMPT",
    "MINIMAL_ATC_PROMPT",
    "EXTENDED_ATC_PROMPT",
    "get_prompt",
    "create_custom_prompt",
    "AVAILABLE_PROMPTS",
]
