"""Pluggable LLM reader for cited, free-form synthesis.

Status in this submission
-------------------------
The **headline evaluation does not use this module.** No API key was configured
for this build, and -- more importantly -- an LLM generator would confound the
experiment: a strong model answers many 2-hop Wikipedia questions from
parametric memory, so a measured gain would mix "better context" with "better
model". The local extractive reader in ``reader.py`` is therefore the fixed
generator for every number in EVAL.md. See ``closed_book`` for how much of this
benchmark is answerable with no retrieval at all.

This module exists so that (a) the synthesis-with-claim-level-citations path is
real, runnable code rather than a description, and (b) the headline table can be
extended with answer-level LLM numbers by setting one env var. It shares the
same ``Answer``/``Claim`` schema and the same
``ctxlayer.synth.cite.verify_citations`` verifier as the extractive path, so a
citation emitted by an LLM is held to the identical mechanical standard: the
anchor must exist and its text must still match the source bytes.

Prompt contract
---------------
The assembled context is rendered with anchor ids inline (see
``AssembledContext.render``), and the model is required to emit JSON claims that
cite those ids. Asking for *claim-level* citations rather than a trailing
reference list is deliberate: it is the only form that can be verified
per-assertion, which is what Hard Requirement #1 asks for. Anything the model
returns is treated as untrusted -- ids are resolved against the knowledge base
and unparseable output falls back to post-hoc attribution.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from ..config import ReaderConfig
from ..schema import Answer, AssembledContext, Claim

__all__ = ["LLMReader", "LLMProvider", "get_provider", "SYSTEM_PROMPT"]


SYSTEM_PROMPT = """You answer questions using ONLY the provided sources.

The context contains passages. Every sentence is prefixed with an anchor id in \
parentheses, like (a1b2c3d4e5f6::p0::s2). These ids are the only valid citations.

Rules:
1. Use ONLY information stated in the context. Never use prior knowledge.
2. Decompose your reasoning into atomic factual claims. Each claim must cite the \
anchor id(s) of the exact sentence(s) that state it.
3. If the context does not contain enough information, set "answer" to null and \
explain in "missing".
4. The final answer must be as short as possible: a name, date, number or short \
phrase copied from the sources.

Respond with ONLY a JSON object:
{
  "claims": [
    {"text": "<one atomic factual claim>", "anchors": ["<anchor_id>", ...]}
  ],
  "answer": "<short final answer, or null>",
  "missing": "<what was missing, or null>"
}"""


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str, cfg: ReaderConfig) -> str: ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str, cfg: ReaderConfig) -> str:
        resp = self._client.messages.create(
            model=cfg.llm_model,
            max_tokens=cfg.llm_max_tokens,
            temperature=cfg.llm_temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    def complete(self, system: str, user: str, cfg: ReaderConfig) -> str:
        resp = self._client.chat.completions.create(
            model=cfg.llm_model,
            max_tokens=cfg.llm_max_tokens,
            temperature=cfg.llm_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str) -> None:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai

    def complete(self, system: str, user: str, cfg: ReaderConfig) -> str:
        model = self._genai.GenerativeModel(
            cfg.llm_model, system_instruction=system
        )
        resp = model.generate_content(
            user,
            generation_config={
                "temperature": cfg.llm_temperature,
                "max_output_tokens": cfg.llm_max_tokens,
            },
        )
        return resp.text or ""


_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def get_provider(cfg: ReaderConfig) -> LLMProvider | None:
    """Return a provider if its key is present, else ``None``.

    Returning ``None`` rather than raising lets the evaluation run end-to-end
    without a key: the runner reports LLM-dependent columns as unavailable
    instead of failing, which is what keeps the repository runnable for a
    reviewer who has no key.
    """
    provider = cfg.llm_provider
    env = _ENV_KEYS.get(provider)
    key = os.environ.get(env or "", "").strip()
    if not key:
        return None
    if provider == "anthropic":
        return AnthropicProvider(key)
    if provider == "openai":
        return OpenAIProvider(key)
    if provider == "gemini":
        return GeminiProvider(key)
    raise ValueError(f"unknown llm_provider: {provider!r}")


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> dict[str, Any] | None:
    """Extract the JSON object from a model response.

    Models wrap JSON in prose or fences often enough that a bare
    ``json.loads`` is not robust; we take the outermost brace-delimited block.
    """
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = _JSON_BLOCK.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class LLMReader:
    def __init__(self, cfg: ReaderConfig, cache: Any | None = None) -> None:
        self.cfg = cfg
        self.provider = get_provider(cfg)
        self.available = self.provider is not None
        # Memoising the *raw completion* matters more here than for the local
        # reader: an uncached re-run costs money and would make a published
        # number depend on when it was produced. The key covers every input the
        # completion is a function of, so a hit is exact, not approximate.
        self.cache = cache

    def _cache_key(self, question: str, context: AssembledContext) -> str | None:
        if self.cache is None or not getattr(self.cache, "enabled", False):
            return None
        return self.cache.key(
            self.cfg.llm_provider,
            self.cfg.llm_model,
            f"{self.cfg.llm_temperature}",
            f"{self.cfg.llm_max_tokens}",
            SYSTEM_PROMPT,
            question,
            context.render(),
        )

    def read(
        self, question: str, context: AssembledContext, focus: str = "unknown"
    ) -> Answer:
        if self.provider is None:
            raise RuntimeError(
                f"No API key for provider {self.cfg.llm_provider!r}. "
                f"Set {_ENV_KEYS.get(self.cfg.llm_provider)} or use "
                "reader.kind='extractive'."
            )
        if not context.passages:
            return Answer(answer="", abstained=True, context=context,
                          meta={"reason": "empty_context"})

        user = f"Context:\n\n{context.render()}\n\nQuestion: {question}"
        ck = self._cache_key(question, context)
        cached = self.cache.get(ck) if ck else None
        if cached is not None:
            raw = cached.get("raw", "")
        else:
            raw = self.provider.complete(SYSTEM_PROMPT, user, self.cfg)
            if ck:
                self.cache.put(ck, {"raw": raw})
        data = _parse(raw)

        if data is None:
            # Unparseable: fall back to post-hoc attribution so the answer is
            # still anchored to real locations rather than silently uncited.
            from .cite import attribute_claims

            text = raw.strip()
            return Answer(
                answer=text,
                claims=attribute_claims_safe(attribute_claims, context, text),
                context=context,
                raw_generation=raw,
                meta={"parse_failed": True, "model": self.cfg.llm_model},
            )

        claims: list[Claim] = []
        for c in data.get("claims") or []:
            if not isinstance(c, dict):
                continue
            anchors = [a for a in (c.get("anchors") or []) if isinstance(a, str)]
            claims.append(Claim(text=str(c.get("text", "")).strip(), anchor_ids=anchors))

        answer = data.get("answer")
        abstained = answer is None or not str(answer).strip()
        return Answer(
            answer="" if abstained else str(answer).strip(),
            claims=claims,
            abstained=abstained,
            context=context,
            raw_generation=raw,
            meta={
                "model": self.cfg.llm_model,
                "provider": self.cfg.llm_provider,
                "missing": data.get("missing"),
            },
        )


def attribute_claims_safe(fn, context: AssembledContext, text: str) -> list[Claim]:
    """``attribute_claims`` needs a KB for semantic scoring; degrade to lexical."""
    try:
        return fn(None, text, context)  # type: ignore[arg-type]
    except Exception:
        return [Claim(text=text, anchor_ids=[])]
