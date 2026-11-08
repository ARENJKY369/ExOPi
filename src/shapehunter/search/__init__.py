"""Search layer: BLS, template bank, matched filtering and model selection."""

from __future__ import annotations

from shapehunter.core import Candidate
from shapehunter.search.bls import BLSResult, default_durations, period_grid, refine_result, run_bls
from shapehunter.search.matched_filter import MatchResult, best_match, match_bank, matched_filter
from shapehunter.search.model_selection import ModelComparison, bic, compare_families
from shapehunter.search.single_event import SingleEvent, event_mask, find_single_events
from shapehunter.search.template_bank import Template, TemplateBank, template_profile

__all__ = [
    "Candidate",
    "BLSResult",
    "run_bls",
    "refine_result",
    "default_durations",
    "period_grid",
    "Template",
    "TemplateBank",
    "template_profile",
    "MatchResult",
    "matched_filter",
    "match_bank",
    "best_match",
    "ModelComparison",
    "compare_families",
    "bic",
    "SingleEvent",
    "find_single_events",
    "event_mask",
]
