"""Synthesis: a fixed, local, deterministic reader with exact provenance.

Why a local extractive reader is the *primary* generator
--------------------------------------------------------
The experiment we need to run is "does this context layer deliver better
context?". If the generator is an LLM, three things go wrong:

* **Credit attribution.** A strong LLM can often answer a 2-hop Wikipedia
  question from parametric memory, so a gain (or a lack of one) is confounded
  with model knowledge. Our closed-book baseline measures exactly this leakage.
* **Determinism.** Evaluating 7 systems x 500 questions x N ablations against a
  sampling API gives numbers that move between runs.
* **Reproducibility and cost.** Anyone re-running the evaluation needs the same
  model version and a funded API key.

So the headline protocol **holds the reader fixed** -- the same local model,
same prompt construction, same decoding -- and varies only the context layer.
Every answer-level delta in EVAL.md is therefore attributable to retrieval and
assembly. An LLM reader is available via ``ReaderConfig.kind='llm'`` for
qualitative demonstration and for the free-form synthesis path, and the
citation machinery is shared by both.

Per-passage reading, and why it gives provenance for free
---------------------------------------------------------
Rather than concatenating the context into one long window, we score
``(question, passage)`` independently for every passage in the assembled
context and take the globally best span. That choice buys three things:

1. **Exact provenance.** The winning span has character offsets inside one
   known passage, so the citing sentence -- and therefore the ``Anchor`` -- is
   determined mechanically, not guessed. A concatenated window would require
   mapping offsets back through truncation and padding.
2. **No position bias.** A long concatenated context makes the answer's
   probability depend on where in the window it landed ("lost in the middle").
   Scoring passages independently removes that confound, so a baseline is not
   penalised for the order its ranking happened to produce.
3. **Calibrated abstention.** SQuAD2-trained models emit a null score per
   passage; the margin between the best span and the best null is a usable
   confidence, which drives abstention.

Honest limitation
-----------------
This reader *extracts*; it does not reason across passages. In MuSiQue the
final answer is a span in the last hop's paragraph, so retrieve-then-extract is
a sound protocol -- but it means answer EM/F1 here measures *"did the context
layer put the answer-bearing paragraph, and its supporting chain, in front of
the reader?"* rather than *"can the reader chain facts?"*. That is the quantity
this assignment is about, and we state it plainly rather than implying the
reader is doing multi-hop inference. EVAL.md reports the oracle row so reading
error and retrieval error can be separated.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np

from ..config import ReaderConfig
from ..index.dense import resolve_device
from ..schema import Anchor, Answer, AssembledContext, Claim, Passage

__all__ = ["ExtractiveReader", "pick_anchor_for_span"]


# Soft answer-type compatibility. A mismatch only *demotes* a candidate; it can
# never remove the only candidate, so a wrong focus guess cannot zero out a
# question.
_DATE_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)
_NUM_RE = re.compile(r"\d")
_TYPE_PENALTY = 0.65


def pick_anchor_for_span(
    passage: Passage, char_start: int, char_end: int
) -> Anchor | None:
    """The anchor whose sentence span contains the extracted answer.

    Overlap-based rather than containment-based: a span can straddle a sentence
    boundary when the splitter and the model disagree, and we still want the
    best-matching sentence instead of no citation at all.
    """
    best, best_overlap = None, 0
    for a in passage.anchors:
        overlap = min(char_end, a.char_end) - max(char_start, a.char_start)
        if overlap > best_overlap:
            best, best_overlap = a, overlap
    return best or (passage.anchors[0] if passage.anchors else None)


class ExtractiveReader:
    _cache: dict[tuple[str, str], tuple[object, object]] = {}

    def __init__(self, cfg: ReaderConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self._tok = None
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        key = (self.cfg.model, self.device)
        if key not in self._cache:
            tok = AutoTokenizer.from_pretrained(self.cfg.model)
            model = (
                AutoModelForQuestionAnswering.from_pretrained(self.cfg.model)
                .to(self.device)
                .eval()
            )
            for p in model.parameters():
                p.requires_grad_(False)
            self._cache[key] = (tok, model)
        self._tok, self._model = self._cache[key]
        self._torch = torch

    # ------------------------------------------------------------- scoring
    def _score_passages(
        self, question: str, passages: Sequence[Passage]
    ) -> list[tuple[float, int, int, float]]:
        """Best span per passage as ``(score, char_start, char_end, null)``."""
        self._ensure()
        torch = self._torch
        tok = self._tok
        results: list[tuple[float, int, int, float]] = []

        for i in range(0, len(passages), self.cfg.batch_size):
            chunk = list(passages[i : i + self.cfg.batch_size])
            enc = tok(  # type: ignore[misc]
                [question] * len(chunk),
                [p.text for p in chunk],
                truncation="only_second",
                max_length=self.cfg.max_length,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")
            seq_ids = [enc.sequence_ids(j) for j in range(len(chunk))]
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                out = self._model(**enc)  # type: ignore[misc]
            starts = out.start_logits.float().cpu().numpy()
            ends = out.end_logits.float().cpu().numpy()

            for j in range(len(chunk)):
                off = offsets[j].numpy()
                sids = seq_ids[j]
                # Only context tokens (sequence id 1) are valid span endpoints.
                valid = np.array(
                    [1.0 if sid == 1 else 0.0 for sid in sids], dtype=np.float32
                )
                s_log = np.where(valid > 0, starts[j], -1e9)
                e_log = np.where(valid > 0, ends[j], -1e9)
                # Null score at the CLS position: the SQuAD2 convention.
                null = float(starts[j][0] + ends[j][0])

                n_best = 20
                s_idx = np.argsort(-s_log)[:n_best]
                e_idx = np.argsort(-e_log)[:n_best]
                best = (-1e18, 0, 0)
                for si in s_idx:
                    for ei in e_idx:
                        if ei < si:
                            continue
                        if ei - si + 1 > self.cfg.max_answer_words * 4:
                            continue
                        sc = float(s_log[si] + e_log[ei])
                        if sc > best[0]:
                            best = (sc, int(si), int(ei))
                if best[0] < -1e17:
                    results.append((-1e9, 0, 0, null))
                    continue
                cs = int(off[best[1]][0])
                ce = int(off[best[2]][1])
                results.append((best[0], cs, ce, null))
        return results

    # ---------------------------------------------------------------- answer
    def read(
        self,
        question: str,
        context: AssembledContext,
        focus: str = "unknown",
    ) -> Answer:
        if not context.passages:
            return Answer(
                answer="",
                abstained=True,
                context=context,
                meta={"reason": "empty_context", "reader": self.cfg.model},
            )

        scored = self._score_passages(question, context.passages)

        # Passage-selection term. The span margin is calibrated *within* a
        # passage (span vs. that passage's own null), not *between* passages, so
        # comparing raw margins across passages lets a confident-looking span in
        # an irrelevant passage beat the correct one -- we observed exactly that.
        # Combining the span score with the retriever's passage score is the
        # standard remedy (cf. DPR, which adds a passage-selection score to the
        # span score). Retrieval scores are min-max normalised per question,
        # since they are not comparable across questions.
        rel = np.asarray(context.scores, dtype=np.float64)
        if rel.size == len(context.passages) and rel.size > 1:
            lo, hi = float(rel.min()), float(rel.max())
            rel_norm = (rel - lo) / (hi - lo) if hi - lo > 1e-9 else np.ones_like(rel)
        else:
            rel_norm = np.ones(len(context.passages), dtype=np.float64)

        best_idx, best_val, best_span = -1, -1e18, (0, 0)
        for i, (score, cs, ce, null) in enumerate(scored):
            if ce <= cs:
                continue
            text = context.passages[i].text[cs:ce]
            if not text.strip():
                continue
            margin = score - null
            if focus in {"date", "number"} and not (
                _DATE_RE.search(text) if focus == "date" else _NUM_RE.search(text)
            ):
                margin *= _TYPE_PENALTY
            combined = margin + self.cfg.passage_selection_weight * float(rel_norm[i])
            if combined > best_val:
                best_idx, best_val, best_span = i, combined, (cs, ce)

        if best_idx < 0:
            return Answer(
                answer="",
                abstained=True,
                context=context,
                meta={"reason": "no_span", "reader": self.cfg.model},
            )

        passage = context.passages[best_idx]
        cs, ce = best_span
        answer_text = passage.text[cs:ce].strip(" .,;:’'\"")
        anchor = pick_anchor_for_span(passage, cs, ce)

        abstain = bool(self.cfg.abstain_threshold) and best_val < self.cfg.abstain_threshold
        claim = Claim(
            text=answer_text,
            anchor_ids=[anchor.anchor_id] if anchor else [],
            support_score=float(best_val),
        )
        return Answer(
            answer="" if abstain else answer_text,
            claims=[claim],
            abstained=abstain,
            context=context,
            raw_generation=answer_text,
            meta={
                "reader": self.cfg.model,
                "margin": float(best_val),
                "passage_id": passage.passage_id,
                "doc_title": passage.doc_title,
                "char_start": cs,
                "char_end": ce,
                "focus": focus,
                # Probability-like confidence for the risk-coverage curve in
                # EVAL.md. Monotone in the margin; not a calibrated posterior.
                "confidence": 1.0 / (1.0 + math.exp(-best_val / 4.0)),
            },
        )
