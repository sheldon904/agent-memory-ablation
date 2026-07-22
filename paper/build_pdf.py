"""Render the paper to a real PDF with reportlab (no TeX toolchain required).

    python -m paper.build_pdf        # -> paper/agent-memory-ablation.pdf

Tables are built from results/summary.json and corpus/manifest.json, so the PDF
always matches the last `make eval`. Figures are the PNGs from
`paper.generate_figures`. This is the dependency-light PDF path; `paper/paper.tex`
remains the arXiv/LaTeX source for venue submission.
"""

from __future__ import annotations

import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageTemplate, Paragraph, Spacer, Table,
    TableStyle, KeepTogether,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "agent-memory-ablation.pdf")

SERIF = "Times-Roman"
SERIF_B = "Times-Bold"
SERIF_I = "Times-Italic"

INK = colors.HexColor("#111111")
RULE = colors.HexColor("#333333")
HEADER_GREY = colors.HexColor("#666666")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _load():
    with open(os.path.join(ROOT, "results", "summary.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    with open(os.path.join(ROOT, "corpus", "manifest.json"), encoding="utf-8") as fh:
        corpus = json.load(fh)
    return data["summary"], data["storage"], data["manifest"], corpus


def pct(x):
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------
# styles
# ---------------------------------------------------------------------------
def styles():
    ss = getSampleStyleSheet()
    S = {}
    S["title"] = ParagraphStyle("title", parent=ss["Title"], fontName=SERIF_B,
                                fontSize=15.5, leading=19, alignment=TA_CENTER,
                                textColor=INK, spaceAfter=6)
    S["author"] = ParagraphStyle("author", fontName=SERIF, fontSize=11, leading=14,
                                 alignment=TA_CENTER, textColor=INK)
    S["meta"] = ParagraphStyle("meta", fontName=SERIF, fontSize=9.5, leading=12,
                               alignment=TA_CENTER, textColor=HEADER_GREY)
    S["abstract_h"] = ParagraphStyle("abh", fontName=SERIF_B, fontSize=10.5,
                                     leading=13, alignment=TA_CENTER, spaceBefore=10,
                                     spaceAfter=4, textColor=INK)
    S["abstract"] = ParagraphStyle("ab", fontName=SERIF, fontSize=9.3, leading=12.4,
                                   alignment=TA_JUSTIFY, textColor=INK,
                                   leftIndent=22, rightIndent=22)
    S["h1"] = ParagraphStyle("h1", fontName=SERIF_B, fontSize=12, leading=15,
                             spaceBefore=12, spaceAfter=4, textColor=INK)
    S["h2"] = ParagraphStyle("h2", fontName=SERIF_B, fontSize=10.5, leading=13,
                             spaceBefore=8, spaceAfter=3, textColor=INK)
    S["body"] = ParagraphStyle("body", fontName=SERIF, fontSize=10, leading=13.2,
                               alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5)
    S["bullet"] = ParagraphStyle("bullet", parent=S["body"], leftIndent=16,
                                 bulletIndent=4, spaceAfter=3)
    S["caption"] = ParagraphStyle("cap", fontName=SERIF, fontSize=8.6, leading=11,
                                  alignment=TA_CENTER, textColor=INK, spaceBefore=3,
                                  spaceAfter=8)
    S["cell"] = ParagraphStyle("cell", fontName=SERIF, fontSize=8.8, leading=11,
                               alignment=TA_CENTER, textColor=INK)
    S["cell_l"] = ParagraphStyle("cell_l", parent=S["cell"], alignment=TA_LEFT)
    S["cell_b"] = ParagraphStyle("cell_b", parent=S["cell"], fontName=SERIF_B)
    S["cell_hdr"] = ParagraphStyle("cell_h", parent=S["cell"], fontName=SERIF_B)
    S["ref"] = ParagraphStyle("ref", fontName=SERIF, fontSize=8.8, leading=11.4,
                              alignment=TA_LEFT, textColor=INK, leftIndent=16,
                              firstLineIndent=-16, spaceAfter=3)
    return S


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=letter,
                         leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                         topMargin=1.0 * inch, bottomMargin=0.85 * inch,
                         title="Agent-Memory Ablation",
                         author="William White")
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id="main")
        self.addPageTemplates([PageTemplate(id="t", frames=[frame],
                                            onPage=self._decorate)])

    def _decorate(self, canvas, doc):
        canvas.saveState()
        w, h = letter
        canvas.setFont(SERIF, 8)
        canvas.setFillColor(HEADER_GREY)
        canvas.drawString(0.9 * inch, h - 0.62 * inch, "Agent-Memory Ablation")
        canvas.drawRightString(w - 0.9 * inch, h - 0.62 * inch,
                               "Sheldon Dynamics")
        canvas.setStrokeColor(colors.HexColor("#BBBBBB"))
        canvas.setLineWidth(0.5)
        canvas.line(0.9 * inch, h - 0.68 * inch, w - 0.9 * inch, h - 0.68 * inch)
        canvas.drawCentredString(w / 2, 0.55 * inch, str(doc.page))
        canvas.restoreState()


def make_table(S, header, rows, col_widths, bold=None):
    """bold: set of (col, row) coords (row 1-indexed over data rows) to embolden."""
    bold = bold or set()
    data = [[Paragraph(h, S["cell_hdr"]) for h in header]]
    for r, row in enumerate(rows, start=1):
        cells = []
        for c, val in enumerate(row):
            st = S["cell_l"] if c == 0 else (S["cell_b"] if (c, r) in bold else S["cell"])
            cells.append(Paragraph(str(val), st))
        data.append(cells)
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.1, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 1.1, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def figure(path, width_in, caption, S, num):
    img = Image(path)
    iw, ih = img.drawWidth, img.drawHeight
    w = width_in * inch
    img.drawWidth, img.drawHeight = w, ih * (w / iw)
    img.hAlign = "CENTER"
    cap = Paragraph(f"<b>Figure {num}.</b> {caption}", S["caption"])
    return KeepTogether([img, cap])


def table_block(t, caption, S, num):
    cap = Paragraph(f"<b>Table {num}.</b> {caption}", S["caption"])
    return KeepTogether([t, Spacer(1, 3), cap])


# ---------------------------------------------------------------------------
# story
# ---------------------------------------------------------------------------
def build_story(S):
    summary, storage, manifest, corpus = _load()
    arms = ["rag", "graph", "hybrid"]
    name = {"rag": "rag", "graph": "graph", "hybrid": "hybrid"}
    st = []

    def P(text):
        st.append(Paragraph(text, S["body"]))

    def H1(text):
        st.append(Paragraph(text, S["h1"]))

    def H2(text):
        st.append(Paragraph(text, S["h2"]))

    def B(text):
        st.append(Paragraph(f"•&nbsp;&nbsp;{text}", S["bullet"]))

    # ---- title block ----
    st.append(Paragraph(
        "Ablating a Hybrid Agent-Memory Architecture:<br/>A Controlled Head-to-Head "
        "of Stateless RAG, a Canonical Knowledge Graph,<br/>and Their Hybrid on a "
        "Frozen Synthetic Corpus", S["title"]))
    st.append(Spacer(1, 4))
    st.append(Paragraph("William White", S["author"]))
    st.append(Paragraph("Sheldon Dynamics", S["meta"]))
    st.append(Paragraph("Correspondence: sheldonwhite888@gmail.com &nbsp;&middot;&nbsp; July 2026", S["meta"]))

    # ---- abstract ----
    st.append(Paragraph("Abstract", S["abstract_h"]))
    st.append(Paragraph(
        "A companion study characterized a five-channel hybrid memory substrate running in "
        "production [2] and closed by naming its single most important untested claim: that "
        "the hybrid actually beats the baselines it subsumes, a comparison that report "
        "explicitly declined to run against itself (&ldquo;no comparison against an external "
        "baseline &hellip; has been run,&rdquo; [2] &sect;12). This study runs exactly that "
        "experiment and turns it on its author. We build a frozen synthetic corpus of "
        f"<b>{corpus['n_facts']:,} operational facts</b> about a fictional field-services "
        f"company, with a fully known ground-truth relation graph ({corpus['n_entities']} "
        f"entities, {corpus['n_edges']:,} edges, a closed vocabulary of "
        f"{len(corpus['facts_by_relation'])} relations over "
        f"{len(corpus['entities_by_type'])} entity types), and put <b>three memory "
        "architectures behind one typed MemoryProvider interface</b>: stateless RAG (MiniLM "
        "in sqlite-vec, top-k only), graph-only (canonical graph, forward-traversal recall, "
        "no vectors), and a hybrid that fuses facts + vectors + graph with weighted "
        "Reciprocal Rank Fusion. All three run against three seeded, frozen query sets built "
        "before any arm was executed: 150 known-item, 30 multi-hop (2-3 joins), and 20 "
        "distractor queries (answer absent; the fabrication-pressure probe). On this clean, "
        "canonical corpus the <b>graph-only arm is the surprise strong performer</b>: it "
        f"ties the field on known-item recall (Recall@5 {pct(summary['graph']['recall@5'])}, "
        f"Recall@20 {pct(summary['graph']['recall@20'])}), wins multi-hop retrieval outright "
        f"({pct(summary['graph']['multihop_success'])} terminal-fact success vs. the "
        f"hybrid&rsquo;s {pct(summary['hybrid']['multihop_success'])} and RAG&rsquo;s "
        f"{pct(summary['rag']['multihop_success'])}), and, most consequentially, "
        "is the <b>only arm that does not fabricate</b>, refusing correctly on "
        f"{pct(summary['graph']['refusal_correct_distractor'])} of distractors against just "
        f"{pct(summary['rag']['refusal_correct_distractor'])} for both RAG and the hybrid. "
        f"The hybrid&rsquo;s entire measured advantage reduces to ranking precision (best MRR "
        f"{summary['hybrid']['mrr']:.3f}, best Recall@1 {pct(summary['hybrid']['recall@1'])}), "
        "for which it pays the most latency and storage, and inherits RAG&rsquo;s fabrication "
        "pressure wholesale. We state four hypotheses up front, report that two are refuted, "
        "inventory every failure, devote a section to <i>where the hybrid loses</i>, and "
        "disclose the corpus properties that make this the graph&rsquo;s best case. Every "
        "number reproduces from <font face='Courier'>make eval</font> on the checked-in seeds.",
        S["abstract"]))

    # ---- 1 Introduction ----
    H1("1&nbsp;&nbsp;Introduction")
    P("The Sheldon Dynamics memory program has produced two <i>characterization</i> studies "
      ", a fourteen-run evaluation of a decisioning substrate against a frozen holdout "
      "[1], and a seventeen-day in-vivo instrumentation of a production hybrid memory "
      "substrate [2]. Both study a <i>single system</i>: they measure what a shipped "
      "architecture does, but neither pits that architecture against the alternatives it "
      "claims to improve on. [2] is explicit and repeated about this being its central "
      "limitation, listing eighth among its threats &ldquo;no comparison against an external "
      "baseline (hosted memory API or plain RAG over raw text) has been run,&rdquo; and "
      "closing its next-steps with &ldquo;an external-baseline experiment (plain RAG over the "
      "same corpus) using the E1 harness unchanged.&rdquo;")
    P("This report is that experiment, generalized from &ldquo;plain RAG vs. the hybrid&rdquo; "
      "to a three-way ablation and pointed back at the author&rsquo;s own design assumptions. "
      "The motivating question is narrow and falsifiable: <i>when you strip a production "
      "hybrid memory down to its constituent postures, pure vectors, pure canonical "
      "graph, and run all three head-to-head on one frozen corpus with one frozen eval "
      "harness, does the hybrid actually earn its complexity, and where does it not?</i> The "
      "scientific value of the study is precisely that it can embarrass its author, and partly "
      "does. A hybrid that swept every metric would be a red flag: it would suggest a "
      "benchmark tuned to the conclusion. Synthetic is a feature, not a concession: unlike a "
      "production store, a synthetic corpus with a generated ground-truth graph is shareable, "
      "re-runnable, and gives every query a machine-checkable answer, including the "
      "distractors whose correct answer is &ldquo;not in memory.&rdquo;")

    st.append(figure(os.path.join(FIG, "fig0_architecture.png"), 5.9,
                     "Study design: a frozen corpus and three frozen query sets drive three "
                     "arms behind one MemoryProvider interface; one harness scores every arm "
                     "and exports OTLP traces.", S, 1))

    # ---- 2 Hypotheses ----
    H1("2&nbsp;&nbsp;Hypotheses")
    P("Stated before results, judged against them in &sect;9.")
    B("<b>H1: Vectors win known-item.</b> The vector-bearing arms (RAG, hybrid) will "
      "lead known-item retrieval; the closed-vocabulary graph, lacking dense semantics, will "
      "trail on lexically-paraphrased single-fact queries.")
    B("<b>H2: Structure wins multi-hop.</b> Graph and hybrid will beat RAG on "
      "multi-hop by a wide margin, because joins require traversal that top-k vector recall "
      "cannot perform.")
    B("<b>H3: Refusal is ordered graph &gt; hybrid &gt; RAG.</b> Graph-only will "
      "refuse most correctly on distractors, RAG will fabricate most, and the hybrid will "
      "land <i>between</i> the two.")
    B("<b>H4: The hybrid does not dominate.</b> It will not win every metric; it will "
      "pay measurable costs in latency, storage-per-fact, and at least one quality axis.")

    # ---- 3 The three arms ----
    H1("3&nbsp;&nbsp;The three arms")
    P("All three implement one typed interface, MemoryProvider: <font face='Courier'>build"
      "(facts)</font> indexes the frozen corpus once; <font face='Courier'>recall(query, k)"
      "</font> returns a ranked list of facts plus a single boolean, <font face='Courier'>"
      "refused</font>, the arm&rsquo;s own decision that it has no confident answer. The "
      "harness derives every metric from that response and the wall-clock time of the call, "
      "so no arm sees ground truth or can special-case the evaluation.")
    P("<b>(a) Stateless RAG.</b> Each corpus fact is one retrieval chunk. Facts are embedded "
      "with all-MiniLM-L6-v2 (384-d, CPU) and stored in a sqlite-vec index; recall is "
      "exact-cosine top-k. There is no graph, no entity linking, no state between queries. A "
      "single cosine floor (0.40, matching the source system&rsquo;s semantic threshold [2]) "
      "is both the relevance cutoff and the entire refusal rule, the only defense "
      "against fabrication.")
    P("<b>(b) Graph-only.</b> The ground-truth relation graph is loaded into a forward "
      "(subject&rarr;object) adjacency structure. A query is linked to seed entities by a "
      "shared lexical linker; recall is a bounded forward breadth-first traversal that "
      "collects each reached node&rsquo;s outgoing facts, ranked by hop distance then "
      "token-Jaccard overlap. No vectors. Forward-only traversal follows every reasoning "
      "chain while avoiding the reverse fan-in explosion through hub nodes, the same "
      "hub-suppression instinct [2] encodes, expressed as edge directionality. The arm "
      "refuses exactly when the query links no entity.")
    P("<b>(c) Hybrid.</b> A ported slice of the five-channel design [2], facts + "
      "vectors + graph, with three grounded channels (BM25 lexical, MiniLM cosine, "
      "graph traversal) combined by weighted Reciprocal Rank Fusion [3], the graph channel "
      "up-weighted 2:1 as the precision channel. The refusal rule is deliberately permissive: "
      "it refuses only when <i>both</i> the vector top is below the floor <i>and</i> the "
      "graph links no seed. &sect;6 reports the consequence.")

    # Table 1: corpus inventory
    inv = [
        ["Facts", f"{corpus['n_facts']:,}", "one NL statement per edge/attribute"],
        ["Entities", f"{corpus['n_entities']}", "people, equipment, sites, clients, vendors, teams, jobs, incidents, +"],
        ["Edges", f"{corpus['n_edges']:,}", "directed, typed"],
        ["Relations", f"{len(corpus['facts_by_relation'])}", "closed vocabulary"],
        ["Entity types", f"{len(corpus['entities_by_type'])}", "closed vocabulary"],
    ]
    t1 = make_table(S, ["Layer", "Count", "Notes"], inv, [1.2 * inch, 0.8 * inch, 3.7 * inch])
    st.append(table_block(t1, "Substrate inventory of the frozen corpus (seed "
                          f"{corpus['seed']}). Jobs dominate row count "
                          f"({corpus['facts_by_category'].get('job', 0)} facts) the way email "
                          "exhaust dominates the production store in [2].", S, 1))

    # Table 2: design constants
    consts = [
        ["Embedder", "model / dim", "all-MiniLM-L6-v2 / 384"],
        ["Vector", "store / search", "sqlite-vec vec0 / exact cosine"],
        ["Vector", "cosine floor (relevance + RAG refusal)", "0.40"],
        ["Graph", "traversal / max hops", "forward BFS / 3"],
        ["Hybrid", "fusion / channel weights (vec:facts:graph)", "weighted RRF, k0=60 / 1:1:2"],
        ["Hybrid", "grounded-refusal rule", "vector < 0.40 AND no graph seed"],
        ["Eval", "known-item cutoffs / multi-hop depth", "{1,5,20} / top-10"],
        ["Seeds", "corpus / query", f"{manifest['corpus_seed']} / {manifest['query_seed']}"],
    ]
    t2 = make_table(S, ["Group", "Constant", "Value"], consts,
                    [0.9 * inch, 2.9 * inch, 1.9 * inch])
    st.append(table_block(t2, "Design constants. Fixed and committed; nothing is tuned per "
                          "query or run.", S, 2))

    # ---- 4 Corpus & queries ----
    H1("4&nbsp;&nbsp;Corpus and query sets")
    P("<font face='Courier'>corpus/generate.py</font> deterministically builds "
      "<b>Meridian Field Services</b>, a fictional industrial field-services company, from a "
      "single seeded RNG. Population sizes are fixed constants, so the corpus size is stable "
      "across machines; the seed controls only the random wiring and phrasing. Every fact is "
      "derived from exactly one graph edge or node attribute, so every fact has a known "
      "subject, relation, object, and set of linked entities. The generator writes "
      "facts.jsonl, graph.json, and a manifest carrying SHA-256 checksums; the committed "
      "corpus is verified against a fresh regeneration in the test suite.")
    P("All three query sets are built from the corpus and its ground-truth graph <b>before "
      "any arm is run</b>, and committed with a seed and checksums. <b>Known-item (150):</b> "
      "a natural, paraphrased question whose answer is one specific fact. <b>Multi-hop "
      "(30):</b> a question requiring 2-3 joins (22 two-hop, 8 three-hop) over forward "
      "chains such as <i>job &rarr; USES &rarr; equipment &rarr; MAINTAINED_BY &rarr; vendor"
      "</i>; ground truth is the terminal fact, the full chain, and the answer entity, and "
      "each chain is verified forward-reachable. <b>Distractor (20):</b> a well-formed "
      "question about an entity <i>not</i> in the corpus; the test suite asserts each links "
      "to zero real entities. The distractors are the sharpest instrument: phrased identically "
      "to answerable queries but naming an absent entity, they are semantically close to real "
      "facts, so a dense retriever finds <i>something</i> and must decide whether to answer.")

    # ---- 5 Results ----
    H1("5&nbsp;&nbsp;Results")
    P("All numbers are produced by <font face='Courier'>make eval</font> with the MiniLM "
      "embedder and are the exact contents of <font face='Courier'>results/tables.md</font> "
      "(auto-generated). Each arm is built once over the corpus, then answers all 200 "
      "queries; known-item is scored at cutoffs {1,5,20} from a depth-20 recall, multi-hop at "
      "top-10, distractors on the refused flag. Latency is wall-clock, host-dependent, "
      "reported as measured. Every recall is emitted as an OpenTelemetry span exported to "
      "<font face='Courier'>traces/</font>.")

    H2("5.1&nbsp;&nbsp;Known-item retrieval")
    def row_ki(a):
        return [name[a], pct(summary[a]["recall@1"]), pct(summary[a]["recall@5"]),
                pct(summary[a]["recall@20"]), f"{summary[a]['mrr']:.3f}"]
    bold_ki = _argbest(summary, arms, {1: "recall@1", 2: "recall@5", 3: "recall@20", 4: "mrr"}, maximize=True)
    t = make_table(S, ["Arm", "Recall@1", "Recall@5", "Recall@20", "MRR"],
                   [row_ki(a) for a in arms],
                   [1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch], bold_ki)
    st.append(table_block(t, "Known-item retrieval (150 queries).", S, 3))
    st.append(figure(os.path.join(FIG, "fig1_known_item.png"), 4.6,
                     "Known-item retrieval; the channels err on different queries.", S, 2))
    P("The first surprise is at the top of the table: <b>pure RAG is the weakest known-item "
      "arm</b>, and by a wide margin at depth (Recall@20 of "
      f"{pct(summary['rag']['recall@20'])} means one gold fact in four never surfaces in the "
      "top 20). The cause is directional confusion in a duplicate-rich store: a query like "
      "&ldquo;Who is Dana Reyes&rsquo;s supervisor?&rdquo; embeds close to every &ldquo;X "
      "reports to Dana Reyes&rdquo; fact where Dana is the <i>object</i>, crowding out the "
      "single fact where she is the subject. The graph arm links the subject and returns its "
      "small outgoing set (Recall@20 = 100%). The hybrid wins ranking precision. <b>H1 is "
      "refuted:</b> on an entity-anchored corpus, structure beats pure semantics even for "
      "single-fact retrieval.")

    H2("5.2&nbsp;&nbsp;Multi-hop retrieval")
    def row_mh(a):
        return [name[a], pct(summary[a]["multihop_success"]),
                pct(summary[a]["multihop_entity_recall"]),
                pct(summary[a]["multihop_chain_coverage"])]
    bold_mh = _argbest(summary, arms, {1: "multihop_success", 2: "multihop_entity_recall", 3: "multihop_chain_coverage"}, maximize=True)
    t = make_table(S, ["Arm", "Terminal-fact success", "Answer-entity recall", "Chain coverage"],
                   [row_mh(a) for a in arms],
                   [1.0 * inch, 1.7 * inch, 1.6 * inch, 1.3 * inch], bold_mh)
    st.append(table_block(t, "Multi-hop retrieval (30 queries). Judged at top-10.", S, 4))
    st.append(figure(os.path.join(FIG, "fig2_multihop.png"), 4.6,
                     "Multi-hop; the graph recalls the chain (coverage), the terminal rank is "
                     "the gap.", S, 3))
    P("<b>H2 is confirmed, sharply.</b> RAG cannot join, it retrieves facts "
      "surface-similar to the question, which name the anchor but never the terminal answer, "
      "so it succeeds on zero of thirty. Graph and hybrid traverse. The nuance is the gap "
      f"between <i>chain coverage</i> ({pct(summary['graph']['multihop_chain_coverage'])} for "
      f"the graph) and <i>terminal success</i> ({pct(summary['graph']['multihop_success'])}): "
      "the graph recalls the reasoning chain but a lexical ranker does not prioritize the "
      "answer within it. The hybrid trails the graph on all three measures despite containing "
      "the graph channel; &sect;6 explains why.")

    H2("5.3&nbsp;&nbsp;Refusal and fabrication")
    def row_dx(a):
        return [name[a], pct(summary[a]["refusal_correct_distractor"]),
                pct(summary[a]["over_refusal_answerable"])]
    bold_dx = _argbest(summary, arms, {1: "refusal_correct_distractor"}, maximize=True)
    t = make_table(S, ["Arm", "Refusal correctness (20 distractors)", "Over-refusal (180 answerable)"],
                   [row_dx(a) for a in arms],
                   [1.1 * inch, 2.5 * inch, 2.2 * inch], bold_dx)
    st.append(table_block(t, "Refusal & fabrication.", S, 5))
    st.append(figure(os.path.join(FIG, "fig3_refusal.png"), 4.0,
                     "Fabrication pressure: refusal on 20 distractors (higher = less "
                     "fabrication).", S, 4))
    P("This is the study&rsquo;s most consequential table. The graph arm <b>never "
      "fabricates</b>: an absent entity links nothing, so it refuses correctly on all 20 "
      "distractors while never over-refusing any of the 180 answerable queries. RAG fabricates "
      f"on {pct(1 - summary['rag']['refusal_correct_distractor'])} of distractors, "
      "phrased like real queries, the nearest fact clears the 0.40 floor and the arm answers "
      "with confident nonsense. And the hybrid, despite carrying the graph channel that would "
      "have refused, <b>fabricates exactly as often as RAG.</b> <b>H3 is refuted:</b> the "
      "hybrid is not intermediate; it is pinned to RAG&rsquo;s fabrication rate.")

    H2("5.4&nbsp;&nbsp;Cost: latency and storage")
    def row_cost(a):
        return [name[a], f"{summary[a]['latency_p50_ms']:.2f}", f"{summary[a]['latency_p95_ms']:.2f}",
                f"{storage[a]['index_bytes']:,}", f"{storage[a]['bytes_per_fact']:.0f}"]
    bold_cost = _argbest_min(summary, storage, arms)
    t = make_table(S, ["Arm", "Latency p50 (ms)", "Latency p95 (ms)", "Index bytes", "Bytes/fact"],
                   [row_cost(a) for a in arms],
                   [0.9 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.0 * inch], bold_cost)
    st.append(table_block(t, "Cost. Latency is the one host- and run-dependent row (a single "
                          "reference run); index bytes and every retrieval metric are exact "
                          "and stable.", S, 6))
    st.append(figure(os.path.join(FIG, "fig4_cost.png"), 6.1,
                     "Ranking quality (MRR) versus the two costs, latency and storage.", S, 5))
    lat_ratio = summary["hybrid"]["latency_p50_ms"] / max(summary["graph"]["latency_p50_ms"], 1e-9)
    byte_ratio = storage["hybrid"]["bytes_per_fact"] / max(storage["graph"]["bytes_per_fact"], 1e-9)
    P(f"The graph arm is <b>~{lat_ratio:.0f}&times; faster</b> at the median than the hybrid "
      f"and uses <b>~{byte_ratio:.0f}&times; less storage per fact</b>, because it holds no "
      "embeddings and touches no embedder on the hot path. The two vector arms are dominated "
      "by the MiniLM query embedding; the hybrid adds BM25 and graph work on top. <b>H4 is "
      "confirmed.</b>")

    # ---- 6 Where the hybrid loses ----
    H1("6&nbsp;&nbsp;Where the hybrid loses")
    P("A hybrid that won everything would be a benchmark artifact. This one loses three ways, "
      "two of them by inheriting the failure modes of the channels it fuses; its <i>only</i> "
      "wins on this corpus are ranking-precision (Recall@1, MRR).")
    P("<b>1. It re-inherits RAG&rsquo;s fabrication (the expensive loss).</b> The hybrid "
      "contains the very channel, graph linking, whose absence signal drives the "
      "graph arm to a perfect 100% on distractors. It still fabricates at RAG&rsquo;s rate "
      "because its grounding rule is a permissive OR: it answers if <i>either</i> the vector "
      "channel clears the floor <i>or</i> the graph links a seed. On a distractor the graph "
      "correctly abstains, but the vector channel clears the floor and the OR lets it speak. "
      "This is the general hazard of naive multi-channel grounding: <b>union-of-evidence "
      "maximizes recall and minimizes refusal</b>, and refusal is where fabrication lives. A "
      "graph-<i>veto</i> rule would recover the 100%, at the cost of over-refusing "
      "vector-answerable queries the graph cannot link; that trade is the single "
      "highest-leverage follow-up (&sect;10).")
    P("<b>2. Fusion dilutes the graph channel on multi-hop (the subtle loss).</b> The hybrid "
      "trails graph-only on every multi-hop measure <i>even with the graph channel up-weighted "
      "2:1</i>. Reciprocal Rank Fusion rewards cross-channel agreement, but a multi-hop "
      "terminal fact is found by exactly one channel, the graph, while the two "
      "high-recall channels agree on shallower, surface-similar facts and outvote it. Weighting "
      "the graph channel higher narrows the gap (unweighted RRF scored 13.3%; the 2:1 weight "
      f"lifts it to {pct(summary['hybrid']['multihop_success'])}) but does not close it. "
      "Fusion, the mechanism that buys the hybrid its known-item ranking win, is the same "
      "mechanism that costs it multi-hop recall.")
    P("<b>3. It is the most expensive arm on every cost axis (the unavoidable loss).</b> "
      "Three materialized indices (vectors + BM25 + graph) make it the largest store, and "
      "three channels plus a query embedding make it the slowest. It is strictly "
      "Pareto-dominated by the graph arm on latency, storage, multi-hop, and refusal "
      "simultaneously; it beats the graph arm only on MRR "
      f"({summary['hybrid']['mrr']:.3f} vs {summary['graph']['mrr']:.3f}) and Recall@1 "
      f"({pct(summary['hybrid']['recall@1'])} vs {pct(summary['graph']['recall@1'])}). The "
      "honest one-line valuation: it buys a few points of Recall@1 and 0.05 of MRR for "
      f"~{lat_ratio:.0f}&times; the latency, ~{byte_ratio:.0f}&times; the storage, and a "
      "90-point refusal regression.")

    # ---- 7 Failure inventory ----
    H1("7&nbsp;&nbsp;Failure inventory")
    B("<b>RAG, directional confusion (known-item).</b> "
      f"{pct(1 - summary['rag']['recall@20'])} of gold facts never reach RAG&rsquo;s top-20; "
      "the dominant mode is subject/object inversion in symmetric-looking relations.")
    B("<b>RAG, no-join (multi-hop).</b> 30/30 terminal misses "
      f"({pct(summary['rag']['multihop_chain_coverage'])} chain coverage); no mechanism to "
      "follow an edge.")
    B("<b>RAG &amp; hybrid, confident fabrication (distractor).</b> 18/20 distractors "
      "answered; the nearest real fact clears the 0.40 floor and neither arm has a structural "
      "absence signal to override it.")
    B("<b>Graph, terminal-rank burial (multi-hop).</b> 14/30 terminal misses despite "
      f"{pct(summary['graph']['multihop_chain_coverage'])} chain coverage; hop-ascending "
      "order front-loads the anchor&rsquo;s shallow facts and Jaccard cannot distinguish the "
      "paraphrased terminal. All 8 three-hop queries fail this way.")
    B("<b>Graph, paraphrase ranking (known-item MRR).</b> MRR "
      f"{summary['graph']['mrr']:.3f} trails the hybrid&rsquo;s {summary['hybrid']['mrr']:.3f} "
      "when the query shares no salient token with the gold fact.")
    B("<b>Hybrid, fusion dilution.</b> See &sect;6.2.")

    # ---- 8 Threats ----
    H1("8&nbsp;&nbsp;Threats to validity")
    P("<b>1. The corpus is the graph&rsquo;s best case, and we say so first.</b> Meridian is "
      "synthetic, fully canonical (one clean id per entity, a closed relation vocabulary, "
      "zero extraction noise), and its queries are entity-anchored. That is precisely the "
      "regime in which entity linking and traversal dominate and dense semantics are least "
      "needed. Real operational memory, the production setting [2] the hybrid was built "
      "for, is the opposite: noisy distilled text, no clean ids in the user&rsquo;s "
      "phrasing, heavy paraphrase and coreference. The single most likely way these results "
      "<i>fail to generalize</i> is that they understate the vector and hybrid arms in exactly "
      "the conditions those arms exist to handle.")
    P("<b>2-7.</b> Known-item queries name the anchor entity, handing graph and hybrid a "
      "free seed. Multi-hop success is a strict top-10 terminal-fact bar that undercounts the "
      "graph&rsquo;s chain coverage. Absolute recall depends on one embedder; latency on one "
      "host. Refusal thresholds are pre-committed, not corpus-optimal (avoiding post-hoc "
      "curve-fitting). The hybrid is one point in a design space (&sect;6 names two variants "
      "that would move its numbers). And there is no LLM in the loop: these are retrieval "
      "metrics, not end-task answer-accuracy metrics.")

    # ---- 9 Discussion ----
    H1("9&nbsp;&nbsp;Discussion")
    P("<b>The hypotheses, revisited.</b> H2 and H4 are confirmed. H1 and H3 are "
      "<b>refuted</b>: pure RAG was the <i>worst</i> known-item arm on an entity-anchored "
      "corpus, and the hybrid did not sit between graph and RAG on refusal, it was "
      "pinned to RAG&rsquo;s fabrication rate. Refuting one&rsquo;s own hypotheses is the "
      "intended outcome of a real ablation.")
    P("<b>What the fabrication result means for hybrids generally.</b> The most transferable "
      "finding is architectural: <b>fusing a high-precision abstaining channel with a "
      "high-recall answering channel under union-of-evidence grounding throws away the "
      "abstention.</b> The graph arm&rsquo;s zero-fabrication property is a design asset a "
      "naive hybrid destroys. Any hybrid that wants both recall and calibrated refusal must "
      "make refusal a <i>veto</i> of the grounded channel, not a <i>vote</i>. The production "
      "hybrid&rsquo;s justification therefore lives in the gap between this corpus and real "
      "operational text, and this study locates that gap rather than papering over it.")

    # ---- 10 Next steps ----
    H1("10&nbsp;&nbsp;Recommended next steps")
    P("In order of leverage: (1) <b>graph-veto refusal</b> in the hybrid, measuring recovered "
      "distractor refusal against induced over-refusal; (2) <b>rank-aware fusion</b> that "
      "preserves a lone deep-hop graph hit; (3) a <b>paraphrase / descriptive-reference query "
      "set</b> that does not name the anchor entity, the single most important "
      "generalization test; (4) a <b>noisy corpus variant</b> (extraction noise, alias drift, "
      "coreference) moving Meridian toward production conditions; (5) an <b>LLM "
      "answer-accuracy layer</b> converting retrieval metrics into task metrics.")

    # ---- 11 Conclusion ----
    H1("11&nbsp;&nbsp;Conclusion")
    P("Three memory architectures, one frozen "
      f"{corpus['n_facts']:,}-fact corpus, 200 seeded queries, one eval harness: on this "
      "clean, canonical corpus the <b>graph-only arm wins or ties most axes</b>, "
      f"Recall@20 {pct(summary['graph']['recall@20'])}, multi-hop "
      f"{pct(summary['graph']['multihop_success'])}, refusal "
      f"{pct(summary['graph']['refusal_correct_distractor'])}, at sub-millisecond latency and "
      f"{storage['graph']['bytes_per_fact']:.0f} bytes/fact, while the <b>hybrid&rsquo;s "
      "measured advantage reduces to ranking precision</b> (Recall@1 "
      f"{pct(summary['hybrid']['recall@1'])}, MRR {summary['hybrid']['mrr']:.3f}), bought at "
      f"~{lat_ratio:.0f}&times; the latency, ~{byte_ratio:.0f}&times; the storage, and a "
      "90-point refusal regression it inherits wholesale from its dense channel. Two of four "
      "pre-registered hypotheses are refuted. The corpus is disclosed as the graph&rsquo;s "
      "most favorable case: this study does not show that hybrids are unnecessary; it shows "
      "that on canonical, entity-anchored memory the specialized graph is hard to beat and "
      "cheap to run, that a naive hybrid throws away the graph&rsquo;s zero-fabrication "
      "property, and that the production hybrid&rsquo;s real justification lives in the "
      "messier corpus this study deliberately did not build, and names as the next one "
      "to.")
    P("<b>Practitioner one-liner.</b> On a clean, canonical operational corpus, a vector-free "
      "knowledge graph beats plain RAG and ties or beats a fused hybrid on almost everything "
      "that matters, recall, multi-hop, and especially refusal (100% vs 10%), at "
      "a fraction of the storage and latency; the hybrid&rsquo;s extra machinery buys ranking "
      "precision and, unless refusal is a veto rather than a vote, silently reinherits "
      "RAG&rsquo;s fabrication.")
    P("<b>Reproducibility.</b> Every number reproduces with "
      "<font face='Courier'>pip install -r requirements.txt</font> then "
      "<font face='Courier'>make eval</font>, which regenerates the frozen corpus and query "
      "sets (verifying committed checksums), builds all three arms, runs the 200 queries, and "
      "writes results/*.csv, results/tables.md, results/summary.json, traces/ (OTLP spans), "
      "and the figures. Retrieval metrics are deterministic given the pinned MiniLM; only "
      "latency is host-dependent. AMA_EMBEDDER=hash runs the study fully offline. A 25-test "
      "suite fails if committed artifacts drift from their seeds.")

    # ---- references ----
    H1("References")
    refs = [
        "[1] W. White. <i>Augmented Operational Decisioning with Ontology-Grounded Local "
        "Agents: A Fourteen-Run Empirical Study Against Real Construction-Operations Data.</i> "
        "Little Bear Foundry Research, May 2026.",
        "[2] W. White. <i>Hybrid Holographic Memory in a Production Personal Agent: A "
        "Seventeen-Day Empirical Characterization of Compounding, Ontology-Gated Recall on "
        "Live Operational Data.</i> Sheldon Dynamics, July 2026.",
        "[3] G. V. Cormack, C. L. A. Clarke, S. B&uuml;ttcher. <i>Reciprocal Rank Fusion "
        "Outperforms Condorcet and Individual Rank Learning Methods.</i> SIGIR 2009.",
        "[4] N. Reimers, I. Gurevych. <i>Sentence-BERT: Sentence Embeddings using Siamese "
        "BERT-Networks.</i> EMNLP 2019.",
        "[5] A. Garcia. <i>sqlite-vec.</i> github.com/asg017/sqlite-vec, 2024.",
        "[6] D. Hofstadter. <i>Analogy as the Core of Cognition.</i> The Analogical Mind, MIT "
        "Press, 2001.",
        "[7] T. Plate. <i>Holographic Reduced Representations.</i> IEEE Transactions on Neural "
        "Networks, 6(3), 1995.",
    ]
    for r in refs:
        st.append(Paragraph(r, S["ref"]))

    return st


def _argbest(summary, arms, col_metric, maximize=True):
    """Return {(col,row)} of the best arm per metric column (row 1-indexed)."""
    bold = set()
    for col, metric in col_metric.items():
        vals = [(summary[a][metric], i) for i, a in enumerate(arms, start=1)]
        best = max(vals)[1] if maximize else min(vals)[1]
        bold.add((col, best))
    return bold


def _argbest_min(summary, storage, arms):
    """Cost table: bold the min for latency p50/p95, index bytes, bytes/fact."""
    bold = set()
    for col, get in {
        1: lambda a: summary[a]["latency_p50_ms"],
        2: lambda a: summary[a]["latency_p95_ms"],
        3: lambda a: storage[a]["index_bytes"],
        4: lambda a: storage[a]["bytes_per_fact"],
    }.items():
        vals = [(get(a), i) for i, a in enumerate(arms, start=1)]
        bold.add((col, min(vals)[1]))
    return bold


def main():
    S = styles()
    doc = Doc(OUT)
    S["width"] = doc.width
    doc.build(build_story(S))
    print(f"wrote {OUT} ({os.path.getsize(OUT):,} bytes)")


if __name__ == "__main__":
    main()
