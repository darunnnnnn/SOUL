"""Build a small hand-written knowledge base for EXPLAINING the mechanism.

The evaluated corpus (data/processed) is derived from MuSiQue and pinned by
sha256; the evaluator refuses to run if it changes. So this store is a
SEPARATE knowledge base under data/demo, built with the same ingestion
primitives (same id scheme, same sentence anchors, same offsets) and read by
the same loader. Nothing here touches the evaluated corpus and nothing here is
ever scored.

    python scripts/build_demo_kb.py
    python -m ctxlayer.cli index --kb data/demo
    python -m ctxlayer.cli ask --kb data/demo --qid demo__2hop_microsoft_university --system bm25

The chain is Microsoft -> Bill Gates -> Harvard University. The Microsoft
document names its founders and mentions no education; the Bill Gates document
names the university and never says "Microsoft", calling it "the software
company he had founded". So on the corpus text alone, no query containing the
word "Microsoft" can reach the Gates document.

WHAT THIS STORE CAN AND CANNOT DEMONSTRATE
------------------------------------------
Use it to explain the problem. Do NOT use it to demonstrate that single-shot
retrieval fails, because here it does not. Measured on this store, for
"Which university did the founder of Microsoft attend?":

    bm25            1/2 gold documents   (ranks "The Founder (2016 film)" first)
    dense           2/2 gold documents
    hybrid_rerank   2/2 gold documents

The lexical half behaves exactly as the argument predicts: it is captured by a
film about a hamburger franchise and never reaches Bill Gates. The dense half
retrieves the Gates document anyway -- ranked second, for a question in which
neither "Gates" nor "Harvard" appears.

That is not retrieval from this corpus. It is knowledge inside the sentence
encoder: Microsoft and Bill Gates are among the most strongly associated
entities on the public web the encoder was trained on, so the bridge is
already present in the embedding and does not have to be discovered. A famous
example cannot show the failure it is meant to illustrate.

This is also why MuSiQue is built from obscure entities, and why the honest
live comparison in the walkthrough uses one: no encoder has memorised that
Graft-De Rijp is in North Holland.

A second, independent caveat: the iterative retriever ranks bridge candidates
by inverse document frequency, which is meaningless at 45 documents. On the
3-hop demo question it therefore bridges on "april" and "altair" and scores
WORSE than the baseline. The method needs corpus scale; this store does not
have it. Do not run `--system ours` here as a demonstration.

So: this store is for showing what documents look like, how content is split
into passages and anchors, and what the evidence chain is. Numbers come from
the real corpus.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctxlayer.ingest.musique import _clean, _mint_passage  # noqa: E402
from ctxlayer.schema import stable_id  # noqa: E402

OUT = ROOT / "data" / "demo"

# --------------------------------------------------------------------------
# The documents.
#
# Two of these are the evidence chain. The rest exist to make retrieval a
# real decision rather than a formality -- in particular `The Founder
# (2016 film)`, which a single-shot retriever ranks highly on the word
# "founder" and which cannot answer anything.
# --------------------------------------------------------------------------

DOCS: list[tuple[str, str]] = [
    # ---- the evidence chain -------------------------------------------------
    # Microsoft names its founders and says nothing about anyone's education.
    # Bill Gates names the university but NEVER mentions Microsoft -- it says
    # "the software company he had founded". That is the whole point: no query
    # containing the word "Microsoft" can retrieve the Gates document, so the
    # second hop is unreachable until the first one has been read.
    ("Microsoft",
     "Microsoft Corporation is an American multinational technology company "
     "headquartered in Redmond, Washington. It was founded by Bill Gates and "
     "Paul Allen on April 4, 1975, to develop and sell BASIC interpreters for "
     "the Altair 8800. It rose to dominate the personal computer operating "
     "system market with MS-DOS and later Windows."),
    ("Bill Gates",
     "William Henry Gates III is an American businessman and philanthropist. "
     "He enrolled at Harvard University in 1973 and left two years later, "
     "before completing his degree, to work full time on the software company "
     "he had founded. He stepped down as its chief executive in 2000 and has "
     "since concentrated on philanthropy."),
    ("Harvard University",
     "Harvard University is a private Ivy League research university in "
     "Cambridge, Massachusetts. Established in 1636, it is the oldest "
     "institution of higher learning in the United States. Its endowment is "
     "the largest of any academic institution in the world."),
    ("Cambridge, Massachusetts",
     "Cambridge is a city in Middlesex County, Massachusetts, directly across "
     "the Charles River from Boston. It is home to two major research "
     "universities and a dense cluster of biotechnology firms. The city was "
     "named in 1638 after the English university town of the same name."),

    # ---- lexical decoys -----------------------------------------------------
    # These rank on the question's own words and answer nothing. The film is
    # the clearest case: it is the best match in the store for "founder" and
    # concerns a hamburger franchise.
    ("The Founder (2016 film)",
     "The Founder is a 2016 American biographical drama film directed by John "
     "Lee Hancock. It follows Ray Kroc, a struggling salesman who joined a "
     "small hamburger operation run by two brothers and built it into a "
     "worldwide franchise. Michael Keaton stars as Kroc."),
    ("Microsoft Research",
     "Microsoft Research is the research subsidiary of Microsoft. Founded in "
     "1991, it maintains laboratories in several countries and collaborates "
     "extensively with universities on computer science, cryptography and "
     "applied mathematics. Its staff publish in the open academic literature."),
    ("Redmond, Washington",
     "Redmond is a city in King County, Washington, east of Seattle across "
     "Lake Washington. It is widely known as the headquarters of a major "
     "software company. The city also hosts a long-running bicycle racing "
     "event at its velodrome."),
    ("Altair 8800",
     "The Altair 8800 is a microcomputer designed in 1974 and sold as a kit "
     "through Popular Electronics magazine. It is widely credited with "
     "starting the personal computer industry. Its BASIC interpreter was "
     "written by two young programmers who went on to found a software "
     "company."),
    ("Paul Allen",
     "Paul Gardner Allen was an American businessman, investor and "
     "philanthropist. He is remembered for his investments in sports teams, "
     "scientific research institutes and the arts, and for funding the Allen "
     "Telescope Array. He died in October 2018."),
    ("Gates Foundation",
     "The Gates Foundation is an American private foundation based in Seattle, "
     "Washington. It is among the largest charitable organisations in the "
     "world, with programmes in global health, development and education. It "
     "was launched in the year 2000."),

    # ---- other companies and their founders ---------------------------------
    # Parallel chains, so "the founder of X attended Y" is a pattern with many
    # candidate completions and the retriever cannot succeed by matching the
    # shape of the question.
    ("Apple Inc.",
     "Apple Inc. is an American technology company headquartered in Cupertino, "
     "California. It was founded in April 1976 by Steve Jobs, Steve Wozniak "
     "and Ronald Wayne to sell the Apple I personal computer kit. It became "
     "the first American company valued above one trillion dollars."),
    ("Steve Jobs",
     "Steven Paul Jobs was an American business magnate and inventor. He "
     "enrolled at Reed College in Portland, Oregon in 1972 and withdrew after "
     "a single semester, continuing to audit classes. He later returned to the "
     "computer company he had started and oversaw the iPod and iPhone."),
    ("Steve Wozniak",
     "Stephen Gary Wozniak is an American electronics engineer. He studied at "
     "the University of California, Berkeley, interrupting his degree to work "
     "at Hewlett-Packard, and completed it years later. He single-handedly "
     "designed the Apple I and Apple II computers."),
    ("Reed College",
     "Reed College is a private liberal arts college in Portland, Oregon. "
     "Founded in 1908, it is known for its demanding academic programme and "
     "for a required senior thesis. It operates one of the few research "
     "nuclear reactors run primarily by undergraduates."),
    ("Amazon (company)",
     "Amazon.com is an American multinational technology and retail company "
     "based in Seattle, Washington. It was founded by Jeff Bezos in July 1994 "
     "and began as an online bookstore. It later expanded into cloud computing "
     "through Amazon Web Services."),
    ("Jeff Bezos",
     "Jeffrey Preston Bezos is an American entrepreneur. He graduated from "
     "Princeton University in 1986 with degrees in electrical engineering and "
     "computer science, and worked in quantitative finance in New York before "
     "moving west to start an online retailer."),
    ("Princeton University",
     "Princeton University is a private Ivy League research university in "
     "Princeton, New Jersey. Chartered in 1746, it is the fourth-oldest "
     "institution of higher education in the United States. It is noted for "
     "its mathematics and physics departments."),
    ("Google",
     "Google LLC is an American technology company specialising in internet "
     "search and online advertising. It was founded in September 1998 by Larry "
     "Page and Sergey Brin, who had met as graduate students, and is "
     "headquartered in Mountain View, California."),
    ("Larry Page",
     "Lawrence Edward Page is an American computer scientist. He received a "
     "bachelor's degree from the University of Michigan and entered the "
     "doctoral programme at Stanford University, where he developed a "
     "link-analysis ranking method with a fellow student."),
    ("Sergey Brin",
     "Sergey Mikhailovich Brin is an American computer scientist born in "
     "Moscow. He took his undergraduate degree at the University of Maryland, "
     "College Park and began doctoral study in computer science in California, "
     "which he left to start a search company."),
    ("Stanford University",
     "Stanford University is a private research university in Stanford, "
     "California, adjacent to Palo Alto. It was founded in 1885 by Leland "
     "Stanford and Jane Stanford in memory of their only child. Its proximity "
     "to Silicon Valley has shaped its engineering programmes."),
    ("Meta Platforms",
     "Meta Platforms is an American technology conglomerate based in Menlo "
     "Park, California. Its principal social network was launched in February "
     "2004 by Mark Zuckerberg and four fellow students. The company was "
     "renamed from Facebook, Inc. in October 2021."),
    ("Mark Zuckerberg",
     "Mark Elliot Zuckerberg is an American technology executive. He began "
     "studying psychology and computer science at Harvard University in 2002 "
     "and left in his sophomore year to run the social network he had built in "
     "his dormitory. He has led the company ever since."),
    ("Oracle Corporation",
     "Oracle Corporation is an American computer technology company best known "
     "for its relational database management system. It was founded in 1977 by "
     "Larry Ellison, Bob Miner and Ed Oates under the name Software "
     "Development Laboratories."),
    ("Larry Ellison",
     "Lawrence Joseph Ellison is an American businessman. He attended the "
     "University of Illinois at Urbana-Champaign and later the University of "
     "Chicago, leaving both without graduating, and moved to California to work "
     "as a programmer."),
    ("Dell",
     "Dell Technologies is an American computer manufacturer headquartered in "
     "Round Rock, Texas. The company was started in 1984 by Michael Dell, who "
     "sold computers built to order directly to customers rather than through "
     "retailers."),
    ("Michael Dell",
     "Michael Saul Dell is an American businessman. He began a pre-medical "
     "course at the University of Texas at Austin and left during his first "
     "year to run the computer business he had started from his dormitory "
     "room. He remains its chief executive."),
    ("Intel",
     "Intel Corporation is an American semiconductor manufacturer based in "
     "Santa Clara, California. It was founded in July 1968 by Robert Noyce and "
     "Gordon Moore, and became the dominant supplier of microprocessors for "
     "personal computers."),
    ("Gordon Moore",
     "Gordon Earle Moore was an American businessman and chemist. He earned a "
     "doctorate in chemistry and physics from the California Institute of "
     "Technology in 1954. He is remembered for an observation about the "
     "doubling rate of transistor counts."),
    ("California Institute of Technology",
     "The California Institute of Technology is a private research university "
     "in Pasadena, California. It was founded in 1891 as a preparatory and "
     "vocational school and became a centre for scientific research. It "
     "manages the Jet Propulsion Laboratory."),
    ("IBM",
     "International Business Machines is an American multinational technology "
     "company headquartered in Armonk, New York. It was incorporated in 1911 "
     "as the Computing-Tabulating-Recording Company and renamed in 1924. It "
     "long dominated the mainframe computer market."),
    ("Hewlett-Packard",
     "Hewlett-Packard was an American information technology company founded "
     "in 1939 by Bill Hewlett and David Packard in a garage in Palo Alto, "
     "California. The garage is often described as the birthplace of Silicon "
     "Valley."),
    ("Sun Microsystems",
     "Sun Microsystems was an American computer company founded in 1982 by "
     "four people who had met in graduate school. It sold workstations and "
     "servers and created the Java programming language. It was acquired in "
     "2010."),

    # ---- universities and places, as further near-misses --------------------
    ("Massachusetts Institute of Technology",
     "The Massachusetts Institute of Technology is a private research "
     "university in Cambridge, Massachusetts. Founded in 1861, it adopted the "
     "European polytechnic model and emphasised laboratory instruction. Its "
     "campus runs along the Charles River."),
    ("Yale University",
     "Yale University is a private Ivy League research university in New "
     "Haven, Connecticut. Founded in 1701, it is the third-oldest institution "
     "of higher education in the United States. Its library system is among "
     "the largest in the world."),
    ("University of Washington",
     "The University of Washington is a public research university in Seattle. "
     "Founded in 1861, it is one of the oldest universities on the west coast "
     "of the United States. Its medical school is highly ranked."),
    ("University of Michigan",
     "The University of Michigan is a public research university in Ann Arbor. "
     "Founded in 1817, it is the oldest university in the state. It is a "
     "founding member of the Association of American Universities."),
    ("University of Oxford",
     "The University of Oxford is a collegiate research university in Oxford, "
     "England. There is evidence of teaching as early as 1096, making it the "
     "oldest university in the English-speaking world. It is made up of "
     "thirty-nine semi-autonomous colleges."),
    ("Seattle",
     "Seattle is a seaport city on the west coast of the United States and the "
     "seat of King County, Washington. It lies between Puget Sound and Lake "
     "Washington. The city is a centre of the aerospace, software and retail "
     "industries."),
    ("Cupertino, California",
     "Cupertino is a city in Santa Clara County, California, in the western "
     "Santa Clara Valley. It is known as the headquarters of a large consumer "
     "electronics company. The city was incorporated in 1955."),
    ("Palo Alto, California",
     "Palo Alto is a charter city in Santa Clara County, California. It is "
     "named after a coast redwood tree and adjoins a major private university. "
     "Several technology companies were started in its garages and spare "
     "rooms."),
    ("Silicon Valley",
     "Silicon Valley is a region in the southern part of the San Francisco Bay "
     "Area, California. The name originally referred to the area's "
     "concentration of silicon chip manufacturers and now denotes the American "
     "technology industry generally."),
    ("BASIC",
     "BASIC is a family of general-purpose programming languages designed for "
     "ease of use. It was created in 1964 at Dartmouth College by John Kemeny "
     "and Thomas Kurtz so that students outside science and mathematics could "
     "use computers."),
    ("MS-DOS",
     "MS-DOS is a discontinued operating system for x86-based personal "
     "computers. It was the most widely used member of the DOS family and the "
     "main operating system for IBM PC compatibles through the 1980s. It was "
     "largely superseded by graphical systems."),
    ("Personal computer",
     "A personal computer is a multi-purpose computer whose size and "
     "capabilities make it usable directly by an end user. Early machines were "
     "sold as kits to electronics hobbyists in the mid-1970s before "
     "pre-assembled models reached a mass market."),
]

# --------------------------------------------------------------------------
# The questions.
#
# Gold labels are written by hand here, which is only acceptable because this
# store is never scored. The 3-hop question exists to show that the loop runs
# more than one bridge.
# --------------------------------------------------------------------------

QUESTIONS: list[dict] = [
    {
        "qid": "demo__2hop_microsoft_university",
        "question": "Which university did the founder of Microsoft attend?",
        "answer": "Harvard University",
        "answer_aliases": ["Harvard"],
        "n_hops": 2,
        "hop_type": "2hop",
        "gold_titles": ["Microsoft", "Bill Gates"],
        "note": "Microsoft names the founder; Bill Gates names the university. "
                "'Bill Gates' appears nowhere in the question.",
    },
    {
        "qid": "demo__3hop_microsoft_university_city",
        "question": "In which city is the university attended by the founder of Microsoft located?",
        "answer": "Cambridge",
        "answer_aliases": ["Cambridge, Massachusetts"],
        "n_hops": 3,
        "hop_type": "3hop1",
        "gold_titles": ["Microsoft", "Bill Gates", "Harvard University"],
        "note": "Three documents, two bridges: founder, then university, then city.",
    },
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    passages = []
    by_title = {}
    for title, raw in DOCS:
        text = _clean(raw)
        p = _mint_passage(title, text, 0, {"dataset": "demo", "handwritten": True})
        passages.append(p)
        by_title[title] = p

    # Every anchor must re-resolve to byte-identical text, exactly as the real
    # corpus does -- otherwise the demo would show a citation guarantee the
    # demo itself does not meet.
    for p in passages:
        for a in p.anchors:
            assert p.text[a.char_start:a.char_end] == a.text, (p.doc_title, a.anchor_id)

    all_pids = [p.passage_id for p in passages]

    with (OUT / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for p in passages:
            fh.write(json.dumps({
                "passage_id": p.passage_id,
                "doc_id": p.doc_id,
                "doc_title": p.doc_title,
                "passage_ord": p.passage_ord,
                "text": p.text,
                "anchors": [vars(a) if not hasattr(a, "__slots__") else
                            {k: getattr(a, k) for k in a.__slots__} for a in p.anchors],
                "source_ref": p.source_ref,
            }, ensure_ascii=False) + "\n")

    with (OUT / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for q in QUESTIONS:
            titles = q["gold_titles"]
            fh.write(json.dumps({
                "qid": q["qid"],
                "question": q["question"],
                "answer": q["answer"],
                "answer_aliases": q["answer_aliases"],
                "n_hops": q["n_hops"],
                "hop_type": q["hop_type"],
                "gold_passage_ids": [by_title[t].passage_id for t in titles],
                "gold_doc_ids": [by_title[t].doc_id for t in titles],
                "gold_doc_titles": titles,
                # The whole store is the candidate pool; there are no distractor
                # pools to simulate at this size.
                "pool_passage_ids": all_pids,
                "decomposition": [],
            }, ensure_ascii=False) + "\n")

    corpus_sha = hashlib.sha256((OUT / "corpus.jsonl").read_bytes()).hexdigest()
    manifest = {
        "dataset": "demo-handwritten",
        "split": "demo",
        "source_file": "scripts/build_demo_kb.py",
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "purpose": "Demonstration only. Never evaluated, never reported, and "
                   "deliberately separate from data/processed so the pinned "
                   "benchmark corpus cannot be contaminated.",
        "selection": {
            "n_source": len(DOCS),
            "n_dropped_single_doc": 0,
            "n_eligible": len(DOCS),
            "n_selected": len(QUESTIONS),
        },
        "selection_params": {"subset_size": len(QUESTIONS), "seed": 0,
                             "min_gold_docs": 2, "stratify_by_hops": False},
        "n_questions": len(QUESTIONS),
        "n_passages": len(passages),
        "n_documents": len({p.doc_id for p in passages}),
        "n_anchors": sum(len(p.anchors) for p in passages),
        "hop_type_hist": {q["hop_type"]: 1 for q in QUESTIONS},
        "gold_docs_per_question_hist": {
            str(len(q["gold_titles"])): 1 for q in QUESTIONS},
        "mean_passage_words": round(
            sum(p.n_words for p in passages) / len(passages), 2),
        "mean_gold_passages": round(
            sum(len(q["gold_titles"]) for q in QUESTIONS) / len(QUESTIONS), 3),
        "mean_gold_docs": round(
            sum(len(q["gold_titles"]) for q in QUESTIONS) / len(QUESTIONS), 3),
        "corpus_sha256": corpus_sha,
        "questions_sha256": hashlib.sha256(
            (OUT / "questions.jsonl").read_bytes()).hexdigest(),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"demo knowledge base -> {OUT.relative_to(ROOT)}")
    print(f"  {manifest['n_documents']} documents, {manifest['n_passages']} passages, "
          f"{manifest['n_anchors']} anchors, {manifest['n_questions']} questions")
    print()
    for title, _ in DOCS:
        p = by_title[title]
        mark = ""
        for q in QUESTIONS:
            if title in q["gold_titles"]:
                mark = "  [gold for %d-hop]" % q["n_hops"] if not mark else "  [gold]"
        print(f"  {p.doc_id}  {title}{mark}")
    print()
    for q in QUESTIONS:
        print(f"  {q['qid']}")
        print(f"    Q: {q['question']}")
        print(f"    A: {q['answer']}   via {' -> '.join(q['gold_titles'])}")
        print(f"    {q['note']}")
    print()
    print("  Build the index for it:")
    print("      python -m ctxlayer.cli index --kb data/demo")
    print()
    print("  NOTE: use this store to EXPLAIN the problem, not to demonstrate")
    print("  that single-shot retrieval fails. Measured here, bm25 gets 1/2")
    print("  gold documents (it ranks a film about a hamburger franchise")
    print("  first) but dense gets 2/2: the sentence encoder has already")
    print("  memorised that Bill Gates founded Microsoft, so the bridge is in")
    print("  the embedding rather than in this corpus. A famous example")
    print("  cannot show the failure it illustrates -- which is why MuSiQue")
    print("  uses obscure entities and why the live comparison does too.")
    print("  See the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
