"""Iterative entity-bridged retrieval -- the core of the context layer.

The failure this fixes
----------------------
A single-shot retriever can only find documents that the *question* describes.
For a multi-hop question the later hops are lexically dark::

    Q: "Who is the spouse of the Green performer?"
    gold 1: "Green (Steve Hillage album)"  -- findable: the question says Green
    gold 2: "Miquette Giraudy"             -- unfindable: the question names
                                              neither Giraudy nor Hillage

No amount of reranking recovers gold 2, because reranking only reorders what
recall proposed, and recall never proposed it. This is a hard ceiling, and it is
why we measure ``hybrid + cross-encoder`` as the baseline to beat: it is the
best a single shot can do, and it still cannot see the second document.

The algorithm
-------------
Round 0 retrieves against the question. Each later round:

1. **Frontier.** Take the highest-scoring passages not yet expanded.
2. **Bridge entity selection.** From those passages' mentions, drop the
   question's own anchor entities (that is where we started) and anything
   already expanded, then rank the rest by
   ``idf(entity) x parent_chain_score``. Rare entities inside good evidence are
   the best bridges. Keep the top ``beam_entities``.
3. **Bridge query.** ``entity + relation_terms`` from the question plan. The
   entity alone retrieves the article *about* it; entity + relation retrieves
   the paragraph that *states* the relation -- where the next answer lives.
4. **Constrained retrieval.** Candidates come from the mention index (exact
   phrase containment, high precision) and are then scored by the full fused
   retriever + cross-encoder *against the bridge query*.
5. **Chain scoring.** Scores from different queries are not comparable, so each
   round's reranker scores are min-max normalised within the round and combined
   with the parent's chain score and the bridge entity's idf (see
   ``IterativeConfig``). A passage is good if it is relevant *and* was reached
   through good evidence.
6. **Adaptive stop.** If the round's best new passage falls below
   ``gain_threshold`` of the running best, stop -- 2-hop questions should not
   pay for 4 hops.

Everything is deterministic and LLM-free, so the gain reported in EVAL.md is
attributable to retrieval structure and not to a language model's reasoning.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..config import SystemConfig
from ..index.bm25 import STOPWORDS
from ..index.store import KnowledgeBase
from ..schema import RetrievalTrace, ScoredPassage
from .decompose import QueryPlan, analyze_question
from .hybrid import SingleShotRetriever

__all__ = ["IterativeRetriever"]


@dataclass
class _PoolEntry:
    passage_id: str
    chain: float
    hop: int
    rel_norm: float
    parent: str | None
    bridge_entity: str | None
    components: dict[str, float] = field(default_factory=dict)


def _minmax_norm(values: list[float]) -> list[float]:
    """Normalise a round's scores to [0, 1].

    Cross-encoder logits are not comparable across queries, so a hop-2 score of
    +3.1 against its bridge query says nothing relative to a hop-0 score of
    +3.1 against the question. Normalising *within the round* makes the
    comparison meaningful, at the cost of discarding absolute confidence --
    which we do not need, since only the ordering feeds assembly.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class IterativeRetriever:
    def __init__(self, kb: KnowledgeBase, cfg: SystemConfig) -> None:
        self.kb = kb
        self.cfg = cfg
        self.single = SingleShotRetriever(kb, cfg)
        # idf normalisation constant for entity rarity
        n = max(1, kb.n_passages)
        self._log_n = math.log(n + 1.0)

    # ------------------------------------------------------------ utilities
    def _entity_idf(self, entity_key: str) -> float:
        """Normalised rarity of an entity in [0, 1]."""
        ents = self.kb.entities
        if ents is None:
            return 0.0
        df = max(1, ents.df(entity_key))
        return min(1.0, math.log((self.kb.n_passages + 1.0) / df) / self._log_n)

    def _select_bridges(
        self,
        frontier: list[_PoolEntry],
        plan: QueryPlan,
        used: set[str],
    ) -> list[tuple[str, float, _PoolEntry]]:
        """Rank candidate bridge entities from the frontier."""
        ents = self.kb.entities
        if ents is None:
            return []
        cfg = self.cfg.iterative
        scored: dict[str, tuple[float, _PoolEntry]] = {}
        for entry in frontier:
            for key in ents.entities_in(entry.passage_id):
                if key in used:
                    continue
                # Skip what the question already told us: those entities got us
                # here, and re-expanding them wastes the beam on hop 1.
                #
                # Stopwords must be stripped before the comparison. Without
                # that, "the lap engine" is not recognised as the question's
                # own entity "Lap Engine" (because of the stray "the") and
                # survives as a bridge. Observed consuming 2 of 4 beam slots
                # with "the lap" and "the lap engine", crowding out the bridge
                # that actually mattered.
                key_toks = {t for t in key.split() if t not in STOPWORDS}
                if key_toks and key_toks <= plan.anchor_entity_tokens:
                    continue
                idf = self._entity_idf(key)
                if idf < cfg.min_entity_idf:
                    continue
                score = idf * max(entry.chain, 1e-3)
                prev = scored.get(key)
                if prev is None or score > prev[0]:
                    scored[key] = (score, entry)
        ranked = sorted(scored.items(), key=lambda kv: (-kv[1][0], kv[0]))
        return [(k, v[0], v[1]) for k, v in ranked[: cfg.beam_entities]]

    # -------------------------------------------------------------- retrieve
    def retrieve(
        self,
        question: str,
        top_k: int = 40,
        restrict_to: list[str] | None = None,
    ) -> tuple[list[ScoredPassage], RetrievalTrace]:
        cfg = self.cfg
        icfg = cfg.iterative
        trace = RetrievalTrace()
        t_start = time.time()
        plan = analyze_question(question)
        trace.timings_ms["plan_ms"] = (time.time() - t_start) * 1000

        pool: dict[str, _PoolEntry] = {}
        expanded: set[str] = set()
        used_entities: set[str] = set()
        best_overall = 0.0

        # ---------------- round 0: the question itself --------------------
        t0 = time.time()
        round0 = self.single.retrieve(
            question, top_k=max(top_k, icfg.per_round_k), restrict_to=restrict_to,
            trace=trace,
        )
        rel = _minmax_norm([c.score for c in round0])
        for c, r in zip(round0, rel):
            pool[c.passage_id] = _PoolEntry(
                passage_id=c.passage_id,
                chain=r,
                hop=0,
                rel_norm=r,
                parent=None,
                bridge_entity=None,
                components=dict(c.components),
            )
        best_overall = max(rel, default=0.0)
        trace.add_round(
            round=0, query=question, n_retrieved=len(round0),
            best_rel=round(best_overall, 4),
            ms=round((time.time() - t0) * 1000, 1),
        )

        if not icfg.enabled:
            return self._finalize(pool, top_k), trace

        # ---------------- rounds 1..max_rounds ----------------------------
        for rnd in range(1, icfg.max_rounds + 1):
            t_round = time.time()
            frontier = sorted(
                (e for e in pool.values() if e.passage_id not in expanded),
                key=lambda e: (-e.chain, e.passage_id),
            )[: icfg.frontier_passages]
            if not frontier:
                trace.add_round(round=rnd, stopped="no_frontier")
                break
            for e in frontier:
                expanded.add(e.passage_id)

            bridges = self._select_bridges(frontier, plan, used_entities)
            if not bridges:
                trace.add_round(round=rnd, stopped="no_bridge_entities")
                break

            round_new: list[tuple[ScoredPassage, _PoolEntry, str, float]] = []
            for entity_key, _ent_score, parent in bridges:
                used_entities.add(entity_key)
                trace.bridge_entities.append(entity_key)

                # Candidate generation: exact phrase containment (+ the article
                # about the entity, when one exists).
                if icfg.use_graph and self.kb.entities is not None:
                    cands = self.kb.entities.candidates_for_entity(
                        entity_key, limit=icfg.candidates_per_entity
                    )
                else:
                    cands = None
                if restrict_to is not None:
                    allowed = set(restrict_to)
                    cands = (
                        [c for c in cands if c in allowed]
                        if cands is not None
                        else list(restrict_to)
                    )
                if cands is not None and not cands:
                    continue

                bq = (
                    plan.bridge_query(entity_key)
                    if icfg.use_query_rewrite
                    else entity_key
                )
                hits = self.single.retrieve(
                    bq,
                    top_k=max(4, icfg.per_round_k // max(1, len(bridges))),
                    restrict_to=cands,
                    rerank_depth=icfg.bridge_rerank_depth,
                    trace=trace,
                )
                for h in hits:
                    # Never follow a bridge back into the passage that supplied
                    # it. Re-retrieving the source of a bridge adds no evidence,
                    # and it was corrupting provenance: a passage could become
                    # its own parent, because expanding an entity found in
                    # passage P frequently returns P itself (it contains the
                    # entity by construction). Observed as
                    # "James Watt <- 'university of glasgow' from James Watt".
                    if h.passage_id == parent.passage_id:
                        continue
                    round_new.append((h, parent, entity_key, self._entity_idf(entity_key)))

            if not round_new:
                # One round's beam yielding nothing must NOT end the search.
                # The frontier has advanced and these entities are now marked
                # used, so the next round selects different -- often far
                # better -- bridges. Breaking here cost a correct answer:
                # round 1 spent its beam on weak entities, returned nothing,
                # and the search stopped before ever expanding "james watt".
                trace.add_round(round=rnd, empty_round=True)
                continue

            # Chain scoring, normalised within this round.
            rel_norm = _minmax_norm([h.score for h, _, _, _ in round_new])
            decay = icfg.hop_decay**rnd
            added = 0
            round_best = 0.0
            for (h, parent, entity_key, idf), rn in zip(round_new, rel_norm):
                chain = decay * (
                    icfg.w_rel * rn
                    + icfg.w_parent * parent.chain
                    + icfg.w_entity * idf
                )
                round_best = max(round_best, chain)
                prev = pool.get(h.passage_id)
                # A passage reachable by several chains keeps its best score.
                if prev is not None and prev.chain >= chain:
                    continue
                comp = dict(h.components)
                comp.update(
                    {"chain": chain, "rel_norm": rn, "entity_idf": idf, "hop": float(rnd)}
                )
                pool[h.passage_id] = _PoolEntry(
                    passage_id=h.passage_id,
                    chain=chain,
                    hop=rnd,
                    rel_norm=rn,
                    parent=parent.passage_id,
                    bridge_entity=entity_key,
                    components=comp,
                )
                # Record how this passage was reached, so synthesis can ground
                # each intermediate hop in the passage that supplied its bridge
                # entity (see synth.cite.build_evidence_chain).
                #
                # Keep the FIRST link, never overwrite it. Score and provenance
                # are different concerns: a passage re-reached later by a
                # higher-scoring chain should take the better score (above) but
                # keep its original, shortest derivation. Overwriting replaced
                # the correct "James Watt <- Lap Engine" link with a later,
                # longer one and destroyed the chain.
                if h.passage_id not in trace.provenance_chain:
                    trace.provenance_chain[h.passage_id] = {
                        "parent": parent.passage_id,
                        "bridge_entity": entity_key,
                        "hop": rnd,
                    }
                added += 1

            trace.add_round(
                round=rnd,
                bridges=[b[0] for b in bridges],
                n_new=added,
                round_best=round(round_best, 4),
                ms=round((time.time() - t_round) * 1000, 1),
            )

            # Adaptive stop: this round found nothing competitive.
            if round_best < icfg.gain_threshold * max(best_overall, 1e-6):
                trace.add_round(round=rnd, stopped="gain_below_threshold")
                best_overall = max(best_overall, round_best)
                break
            best_overall = max(best_overall, round_best)

        trace.timings_ms["total_ms"] = (time.time() - t_start) * 1000
        return self._finalize(pool, top_k), trace

    # --------------------------------------------------------------- output
    def _finalize(self, pool: dict[str, _PoolEntry], top_k: int) -> list[ScoredPassage]:
        entries = sorted(pool.values(), key=lambda e: (-e.chain, e.passage_id))
        out: list[ScoredPassage] = []
        for e in entries[:top_k]:
            comp = dict(e.components)
            comp["chain"] = e.chain
            out.append(
                ScoredPassage(
                    passage_id=e.passage_id,
                    score=e.chain,
                    components=comp,
                    hop=e.hop,
                    reason=(
                        f"hop{e.hop}"
                        + (f" via '{e.bridge_entity}'" if e.bridge_entity else "")
                    ),
                )
            )
        return out
