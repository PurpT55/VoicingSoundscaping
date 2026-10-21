"""Emotion vocabulary, backend mappings and coverage tracking."""

from .mapping import Blend, EmotionRef, Prosody, prosody
from .taxonomy import EMOTIONS, NONVERBALS, Emotion, Nonverbal, find, find_nonverbal, get, get_nonverbal

__all__ = [
    "Blend", "EmotionRef", "Prosody", "prosody", "EMOTIONS", "NONVERBALS", "Emotion", "Nonverbal",
    "find", "find_nonverbal", "get", "get_nonverbal",
]
