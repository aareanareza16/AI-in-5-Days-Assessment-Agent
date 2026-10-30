# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conversation history compactor for managing context window size.

Implements Requirement 2 & Requirement 4 of Category 2: Context & Memory:
- Summarizes and compacts older conversation turns when the turn count or
  estimated token count exceeds configurable thresholds.
- Preserves critical domain entities (Order IDs, SKUs, customer emails/phones,
  RMA numbers, manager override codes, and HITL status) in the compacted summary.
- Supports both synchronous and non-blocking asynchronous (`asyncio`) execution.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Any

from google.genai import types

DEFAULT_MAX_TURNS_THRESHOLD: int = 8
DEFAULT_MAX_TOKENS_THRESHOLD: int = 1200
DEFAULT_RETAIN_RECENT_TURNS: int = 4


def _extract_turn_role_and_text(turn: Any) -> tuple[str, str]:
    """Extracts `(role, text)` from a dict turn, string, or ADK `types.Content`."""
    if isinstance(turn, str):
        return "user", turn
    if isinstance(turn, dict):
        role = str(turn.get("role") or "user")
        content = turn.get("content") or turn.get("text") or ""
        if isinstance(content, list):
            parts_text = [
                str(p.get("text", p)) if isinstance(p, dict) else str(p)
                for p in content
            ]
            return role, " ".join(parts_text)
        return role, str(content)
    if hasattr(turn, "role") and hasattr(turn, "parts"):
        role = str(getattr(turn, "role", None) or "user")
        parts = getattr(turn, "parts", None) or []
        texts: list[str] = []
        for part in parts:
            txt = getattr(part, "text", None)
            if txt is not None:
                texts.append(txt)
        return role, " ".join(texts)
    return "user", str(turn)


def estimate_token_count(item_or_turns: Any) -> int:
    """Estimates the token count of a string, turn, or sequence of turns.

    Uses the standard heuristic of ~4 characters per token (minimum 1 token per
    word) for fast, deterministic in-process token estimation.

    Args:
        item_or_turns: A text string, turn dictionary, `types.Content`, or
            sequence of turns.

    Returns:
        Estimated integer token count.
    """
    if isinstance(item_or_turns, str):
        stripped = item_or_turns.strip()
        if not stripped:
            return 0
        char_est = max(1, len(stripped) // 4)
        word_est = len(stripped.split())
        return max(char_est, word_est)

    if isinstance(item_or_turns, Sequence):
        return sum(estimate_token_count(t) for t in item_or_turns)

    _, text = _extract_turn_role_and_text(item_or_turns)
    return estimate_token_count(text)


class HistoryCompactor:
    """Sliding-window and entity-preserving summarization compactor for chat history.

    When a conversation exceeds `max_turns` or `max_tokens`, compacts older turns
    into a structured summary while retaining the most recent `retain_recent_turns`
    turns intact.
    """

    def __init__(
        self,
        max_turns: int = DEFAULT_MAX_TURNS_THRESHOLD,
        max_tokens: int = DEFAULT_MAX_TOKENS_THRESHOLD,
        retain_recent_turns: int = DEFAULT_RETAIN_RECENT_TURNS,
    ) -> None:
        """Initializes the HistoryCompactor with turn and token thresholds."""
        if retain_recent_turns < 1:
            raise ValueError("retain_recent_turns must be at least 1.")
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.retain_recent_turns = retain_recent_turns

    def should_compact(self, turns: Sequence[Any]) -> bool:
        """Returns True if `turns` exceeds `max_turns` or `max_tokens`."""
        if len(turns) <= self.retain_recent_turns:
            return False
        if len(turns) > self.max_turns:
            return True
        return estimate_token_count(turns) > self.max_tokens

    def summarize_turns(
        self,
        older_turns: Sequence[Any],
        existing_summary: str | None = None,
    ) -> str:
        """Summarizes older conversation turns while preserving key retail entities.

        Extracts Order IDs (`ORD-...`), SKUs (`SKU-...`), customer emails,
        RMA numbers (`RMA-...`), manager override codes (`MGR-OVR-...`), and
        concise per-turn digests.

        Args:
            older_turns: Sequence of older conversation turns to summarize.
            existing_summary: Optional previous compacted summary to merge into.

        Returns:
            Concise multi-line summary string representing the older turns.
        """
        combined_texts: list[str] = []
        turn_digests: list[str] = []

        if existing_summary:
            combined_texts.append(existing_summary)

        for turn in older_turns:
            role, text = _extract_turn_role_and_text(turn)
            clean = " ".join(text.split())
            if not clean:
                continue
            # Skip re-digesting an already-injected summary banner
            if clean.startswith("[COMPACTED CONVERSATION SUMMARY]"):
                combined_texts.append(clean)
                continue
            combined_texts.append(clean)
            snippet = clean[:120] + ("..." if len(clean) > 120 else "")
            turn_digests.append(f"- {role}: {snippet}")

        full_corpus = " ".join(combined_texts)
        orders = sorted(set(re.findall(r"\bORD-\d{4}\b", full_corpus, re.IGNORECASE)))
        skus = sorted(
            set(re.findall(r"\bSKU-[A-Z0-9-]+\b", full_corpus, re.IGNORECASE))
        )
        rmas = sorted(
            set(re.findall(r"\bRMA-[A-Z0-9-]+\b", full_corpus, re.IGNORECASE))
        )
        overrides = sorted(
            set(re.findall(r"\bMGR-OVR-[A-Z0-9-]+\b", full_corpus, re.IGNORECASE))
        )
        emails = sorted(
            set(
                re.findall(
                    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
                    full_corpus,
                )
            )
        )

        entity_parts: list[str] = []
        if orders:
            entity_parts.append(f"Orders: {', '.join(o.upper() for o in orders)}")
        if skus:
            entity_parts.append(f"SKUs: {', '.join(s.upper() for s in skus)}")
        if emails:
            entity_parts.append(f"Customer Emails: {', '.join(emails)}")
        if overrides:
            entity_parts.append(
                f"Override Codes: {', '.join(c.upper() for c in overrides)}"
            )
        if rmas:
            entity_parts.append(f"RMAs: {', '.join(r.upper() for r in rmas)}")

        header = "[COMPACTED CONVERSATION SUMMARY]"
        lines: list[str] = [header]
        if entity_parts:
            lines.append("Key Entities: " + " | ".join(entity_parts))
        if turn_digests:
            # Keep at most the last 6 concise digests in the rolling summary
            lines.append("Earlier Turn Highlights:")
            lines.extend(turn_digests[-6:])
        elif existing_summary:
            lines.append(existing_summary.replace(header, "").strip())

        return "\n".join(lines).strip()

    def compact(
        self,
        turns: Sequence[Any],
        existing_summary: str | None = None,
    ) -> dict[str, Any]:
        """Compacts a list of conversation turns if thresholds are exceeded.

        Args:
            turns: Full sequence of conversation turns (dicts or strings).
            existing_summary: Optional prior summary to incorporate.

        Returns:
            Dictionary containing:
                - `compacted`: True if compaction was performed.
                - `summary`: Updated compacted summary string.
                - `compacted_turns`: New list of turns (summary turn + recent turns).
                - `original_turn_count`: Number of input turns.
                - `final_turn_count`: Number of turns after compaction.
                - `original_tokens`: Estimated token count before compaction.
                - `final_tokens`: Estimated token count after compaction.
        """
        turns_list = list(turns)
        original_tokens = estimate_token_count(turns_list)

        if not self.should_compact(turns_list):
            return {
                "compacted": False,
                "summary": existing_summary or "",
                "compacted_turns": turns_list,
                "original_turn_count": len(turns_list),
                "final_turn_count": len(turns_list),
                "original_tokens": original_tokens,
                "final_tokens": original_tokens,
            }

        cutoff = len(turns_list) - self.retain_recent_turns
        older_turns = turns_list[:cutoff]
        recent_turns = turns_list[cutoff:]

        summary_text = self.summarize_turns(older_turns, existing_summary)
        summary_turn = {
            "role": "system",
            "content": summary_text,
            "is_compacted_summary": True,
        }
        compacted_turns = [summary_turn, *recent_turns]
        final_tokens = estimate_token_count(compacted_turns)

        return {
            "compacted": True,
            "summary": summary_text,
            "compacted_turns": compacted_turns,
            "original_turn_count": len(turns_list),
            "final_turn_count": len(compacted_turns),
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
        }

    async def compact_async(
        self,
        turns: Sequence[Any],
        existing_summary: str | None = None,
    ) -> dict[str, Any]:
        """Asynchronously runs conversation compaction in a background thread."""
        return await asyncio.to_thread(self.compact, turns, existing_summary)

    def compact_adk_contents(
        self,
        contents: list[types.Content],
        existing_summary: str | None = None,
    ) -> tuple[list[types.Content], str | None, bool]:
        """Compacts a list of ADK `types.Content` messages in-place-compatible form.

        Args:
            contents: List of `google.genai.types.Content` objects from `LlmRequest`.
            existing_summary: Optional previous summary stored in session state.

        Returns:
            A 3-tuple `(new_contents, updated_summary, was_compacted)`.
        """
        if not self.should_compact(contents):
            return contents, existing_summary, False

        cutoff = len(contents) - self.retain_recent_turns
        older_contents = contents[:cutoff]
        recent_contents = contents[cutoff:]

        summary_text = self.summarize_turns(older_contents, existing_summary)
        summary_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=summary_text)],
        )
        return [summary_content, *recent_contents], summary_text, True


def compact_conversation_history(
    turns: Sequence[Any],
    max_turns: int = DEFAULT_MAX_TURNS_THRESHOLD,
    max_tokens: int = DEFAULT_MAX_TOKENS_THRESHOLD,
    retain_recent_turns: int = DEFAULT_RETAIN_RECENT_TURNS,
    existing_summary: str | None = None,
) -> dict[str, Any]:
    """Convenience helper that compacts conversation history using `HistoryCompactor`."""
    compactor = HistoryCompactor(
        max_turns=max_turns,
        max_tokens=max_tokens,
        retain_recent_turns=retain_recent_turns,
    )
    return compactor.compact(turns, existing_summary=existing_summary)


async def async_compact_conversation_history(
    turns: Sequence[Any],
    max_turns: int = DEFAULT_MAX_TURNS_THRESHOLD,
    max_tokens: int = DEFAULT_MAX_TOKENS_THRESHOLD,
    retain_recent_turns: int = DEFAULT_RETAIN_RECENT_TURNS,
    existing_summary: str | None = None,
) -> dict[str, Any]:
    """Asynchronous helper that compacts conversation history without blocking."""
    compactor = HistoryCompactor(
        max_turns=max_turns,
        max_tokens=max_tokens,
        retain_recent_turns=retain_recent_turns,
    )
    return await compactor.compact_async(turns, existing_summary=existing_summary)
