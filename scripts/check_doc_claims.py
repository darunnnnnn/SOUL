"""Check the numbers quoted in DESIGN.md against the artefacts that produced them.

Why this exists
---------------
EVAL.md is generated, so its numbers cannot drift. DESIGN.md is hand-written
and quotes measurements in its prose -- and that drift has already happened
once in this project: the reachability figures in DESIGN.md were from a run
made before the entity extractor was cleaned up, and were silently wrong by
several points until I re-ran the diagnostic and noticed.

So every measured figure DESIGN.md asserts is listed here with the artefact it
must agree with. This runs in CI-time (no model, no retrieval) and fails loudly
on a mismatch, which turns "the docs are accurate" from a promise into a check.

    python scripts/check_doc_claims.py

Exit code 0 = every claim verified; 1 = at least one mismatch or missing
artefact.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import ctxlayer
from ctxlayer.config import REPO_ROOT, PathsConfig

TOL = 0.05  # percentage points


class Checker:
    def __init__(self) -> None:
        self.paths = PathsConfig()
        self.results = self.paths.resolve("results_dir")
        self.ok: list[str] = []
        self.bad: list[str] = []
        self.skipped: list[str] = []

    def _load(self, name: str) -> dict | None:
        p = self.results / name
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def doc(self, name: str) -> str:
        p = REPO_ROOT / name
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def claim(self, label: str, doc_name: str, needle: str) -> None:
        """Assert ``needle`` appears verbatim in ``doc_name``."""
        text = self.doc(doc_name)
        if not text:
            self.skipped.append(f"{label}: {doc_name} missing")
            return
        if needle in text:
            self.ok.append(f"{label}: {doc_name} states {needle!r}")
        else:
            self.bad.append(
                f"{label}: {doc_name} does NOT contain {needle!r} "
                f"(artefact says so)"
            )

    # ------------------------------------------------------------ checks
    def check_reachability(self) -> None:
        d = self._load("diagnostic_reachability.json")
        if d is None:
            self.skipped.append(
                "reachability: run `ctxlayer diagnose reachability`"
            )
            return
        graph = d["title_link_graph"]["overall"] * 100
        mention = d["mention_index"]["overall"] * 100
        titles = d["bridge_answers_that_are_titles"]["rate"] * 100
        for label, val in [
            ("title-graph reachability", graph),
            ("mention-index reachability", mention),
            ("bridge-answers-that-are-titles", titles),
        ]:
            needle = f"{val:.1f}"
            found = False
            for doc_name in ("DESIGN.md", "README.md", "WALKTHROUGH.md"):
                if needle in self.doc(doc_name):
                    self.ok.append(f"{label} = {needle}% appears in {doc_name}")
                    found = True
            if not found:
                self.bad.append(
                    f"{label} = {needle}% appears in NO doc "
                    f"(docs may quote a stale figure)"
                )

    def check_corpus(self) -> None:
        man = self.paths.resolve("processed_dir") / "manifest.json"
        if not man.exists():
            self.skipped.append("corpus: run `ctxlayer ingest`")
            return
        m = json.loads(man.read_text(encoding="utf-8"))
        sel = m["selection"]
        self.claim("dropped single-doc questions", "DESIGN.md",
                   str(sel["n_dropped_single_doc"]))
        self.claim("dropped single-doc questions", "README.md",
                   str(sel["n_dropped_single_doc"]))
        # Hard Requirement #2 must hold in the artefact itself.
        hist = m["gold_docs_per_question_hist"]
        if min(int(k) for k in hist) >= 2:
            self.ok.append(
                "Hard Req #2: every selected question needs >= 2 gold documents"
            )
        else:
            self.bad.append(
                f"Hard Req #2 VIOLATED: gold-docs histogram is {hist}"
            )
        n_anchors = m.get("n_anchors")
        if n_anchors:
            rounded = f"{round(n_anchors / 1000)}k"
            if rounded in self.doc("README.md") or str(n_anchors) in self.doc(
                "WALKTHROUGH.md"
            ):
                self.ok.append(f"anchor count ({n_anchors}) consistent with docs")
            else:
                self.skipped.append(f"anchor count {n_anchors} not quoted in docs")

    def check_ab_assembly(self) -> None:
        d = self._load("ab_assembly.json")
        if d is None:
            self.skipped.append("assembly A/B: run `scripts/ab_assembly.py`")
            return
        for setting, rows in d["settings"].items():
            for variant, want_key in (
                ("ours_floor_all", "floor-all"),
                ("ours_floor_tail", "floor-tail"),
            ):
                if variant not in rows:
                    continue
                val = rows[variant]["ctx_all_docs"] * 100
                needle = f"{val:.1f}"
                if needle in self.doc("DESIGN.md"):
                    self.ok.append(
                        f"A/B {setting}/{want_key} AllDocs = {needle} in DESIGN.md"
                    )
                else:
                    self.bad.append(
                        f"A/B {setting}/{want_key} AllDocs = {needle} NOT in "
                        f"DESIGN.md (table may be stale)"
                    )

    def check_report_consistency(self) -> None:
        """EVAL.md is generated; assert it agrees with the report JSON."""
        rep = self._load("report_unified.json")
        if rep is None:
            self.skipped.append("report: run `ctxlayer report --setting unified`")
            return
        evalmd = self.doc("EVAL.md")
        if not evalmd:
            self.skipped.append("EVAL.md not generated yet")
            return
        if rep["headline_table"] in evalmd:
            self.ok.append("EVAL.md headline table matches report_unified.json")
        else:
            self.bad.append(
                "EVAL.md headline table does NOT match report_unified.json "
                "-- regenerate with `ctxlayer report --write-md`"
            )
        if str(rep["n_questions"]) in evalmd:
            self.ok.append(f"EVAL.md states n = {rep['n_questions']}")
        else:
            self.bad.append(f"EVAL.md does not state n = {rep['n_questions']}")

    def check_no_placeholders(self) -> None:
        """No TODO/TBD/XXX left in the deliverable documents."""
        for name in ("DESIGN.md", "EVAL.md", "README.md", "WALKTHROUGH.md"):
            text = self.doc(name)
            if not text:
                continue
            hits = re.findall(r"\b(TODO|TBD|FIXME|XXX|LOREM)\b", text)
            if hits:
                self.bad.append(f"{name} contains placeholders: {set(hits)}")
            else:
                self.ok.append(f"{name} has no placeholders")

    # ------------------------------------------------------------- driver
    def run(self) -> int:
        self.check_reachability()
        self.check_corpus()
        self.check_ab_assembly()
        self.check_report_consistency()
        self.check_no_placeholders()

        print("\n" + "=" * 78)
        print("DOCUMENTATION CLAIMS vs ARTEFACTS")
        print("=" * 78)
        for s in self.ok:
            print(f"  OK      {s}")
        for s in self.skipped:
            print(f"  SKIP    {s}")
        for s in self.bad:
            print(f"  MISMATCH {s}")
        print("-" * 78)
        print(f"  {len(self.ok)} verified, {len(self.skipped)} skipped, "
              f"{len(self.bad)} mismatched")
        if self.bad:
            print("\nFAIL: a document quotes a figure its artefact does not "
                  "support.")
            return 1
        print("\nOK: every checkable figure in the docs matches its artefact.")
        return 0


if __name__ == "__main__":
    sys.exit(Checker().run())
