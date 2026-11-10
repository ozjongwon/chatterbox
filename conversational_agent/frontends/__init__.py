"""Interchangeable frontend adapters."""

from __future__ import annotations

from conversational_agent.frontends.text import TextFrontend
from conversational_agent.frontends.voice import STTResult, VoiceFrontend

__all__ = ['STTResult', 'TextFrontend', 'VoiceFrontend']
