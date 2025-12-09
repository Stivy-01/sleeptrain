"""
Groq-based teacher replacement for the Gemini teacher.

Provides a lightweight wrapper around the Groq API (Llama 3.1 70B) plus a
Gemini-compatible `generate_content` helper so existing hippocampus flows
continue to work.
"""

import json
import os
from typing import Dict, List, Optional

try:
    from groq import Groq
except ImportError as exc:
    Groq = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_groq():
    """Raise a helpful error when groq is not installed."""
    if Groq is None:
        raise ImportError(
            "The 'groq' package is required for GroqTeacher. "
            "Install with `pip install groq`."
        ) from _IMPORT_ERROR


class _GroqResponse:
    """Minimal response shim to mirror .text access expected by Hippocampus."""

    def __init__(self, text: str):
        self.text = text


class GroqTeacher:
    """
    Groq-backed teacher using Llama-3.1-70B for verification and explanations.

    Exposes:
    - generate_explanation(prompt): return plain text completion
    - generate_content(prompt): return object with .text (Gemini-like)
    - hippocampus_scan / generate_cot_dream / score_multi_head: helpers mirroring
      the previous TeacherBrain API where useful.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.1-70b-versatile",
        temperature: float = 0.2,
    ):
        _require_groq()

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is required. Set the env var or pass api_key explicitly."
            )

        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature

        # Lazy import to avoid circulars when utils is imported elsewhere
        try:
            from scripts.others.utils import format_conversation  # type: ignore
        except ImportError:
            # Fallback for running from scripts/others without package context
            from utils import format_conversation  # type: ignore
        self._format_conversation = format_conversation

    def _chat(self, prompt: str, max_tokens: int = 512) -> str:
        """Single-turn chat wrapper."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def generate_explanation(self, prompt: str, max_tokens: int = 512) -> str:
        """Return plain text completion for a given prompt."""
        return self._chat(prompt, max_tokens=max_tokens)

    def generate_content(self, prompt: str, max_tokens: int = 512) -> _GroqResponse:
        """
        Gemini-compatible helper used by hippocampus flows.

        Returns an object with a .text attribute.
        """
        text = self.generate_explanation(prompt, max_tokens=max_tokens)
        return _GroqResponse(text=text)

    # --- Compatibility helpers (mirroring TeacherBrain API) ---
    def hippocampus_scan(self, chat_logs: List[Dict[str, str]]) -> Dict:
        """Score a conversation for memory importance."""
        conversation_text = self._format_conversation(chat_logs)
        prompt = f"""
Analyze this conversation. Rate its importance for long-term memory integration from 1-10.
- 1-3: Small talk, greetings, transient info (Ignore).
- 4-7: General context, preferences.
- 8-10: Critical user facts, identity info, or complex corrections (Must Dream).

Return JSON only: {{"score": int, "reason": "string"}}

Conversation:
{conversation_text}
""".strip()

        response_text = self.generate_explanation(prompt, max_tokens=256)
        return self._safe_json(response_text, default={"score": 0, "reason": "ParseError"})

    def generate_cot_dream(self, chat_logs: List[Dict[str, str]]) -> str:
        """Generate a Chain-of-Thought style dream for consolidation."""
        conversation_text = self._format_conversation(chat_logs)
        prompt = f"""
You are a memory manager for an AI.
The user just provided key identity details.

Write a NATURAL response that answers the question "Who am I and what do you know about me?".
The response should:
1. Be written in the first person ("I know that you are...").
2. Explicitly state the user's name and details.
3. Explain the implication (e.g., "Since you are a Python Architect, I will focus on...")

Do not include <thought> tags. Just give the clear, perfect memory response.

Conversation Context:
{conversation_text}
""".strip()

        return self.generate_explanation(prompt, max_tokens=512)

    def score_multi_head(
        self,
        chat_logs: List[Dict[str, str]],
        existing_embeddings: Optional[List] = None,  # kept for signature compatibility
    ) -> Dict:
        """Multi-head scoring for novelty/utility/importance/privacy."""
        conversation_text = self._format_conversation(chat_logs)
        prompt = f"""
Analyze this conversation across multiple dimensions.
Rate each dimension 1-10:

1. SALIENCE: How novel/unique is this information? (vs generic small talk)
2. UTILITY: How likely is this to be useful in future conversations?
3. IMPORTANCE: How critical is this info (identity, corrections, key facts)?
4. PRIVACY_RISK: How sensitive is this content? (PII, health, financial = high)

Return JSON only:
{{
    "salience": int,
    "utility": int,
    "importance": int,
    "privacy_risk": int,
    "combined_score": int,
    "should_dream": boolean,
    "reason": "string"
}}

Conversation:
{conversation_text}
""".strip()

        response_text = self.generate_explanation(prompt, max_tokens=256)
        defaults = {
            "salience": 5,
            "utility": 5,
            "importance": 5,
            "privacy_risk": 3,
            "combined_score": 5,
            "should_dream": False,
            "reason": "Default",
        }
        return self._safe_json(response_text, default=defaults, ensure_keys=defaults)

    # --- Helpers ---
    @staticmethod
    def _safe_json(
        text: str,
        default: Dict,
        ensure_keys: Optional[Dict] = None,
    ) -> Dict:
        """Best-effort JSON parsing with defaults."""
        clean = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            data = default.copy()

        if ensure_keys:
            for key, value in ensure_keys.items():
                data.setdefault(key, value)
        return data


__all__ = ["GroqTeacher"]
