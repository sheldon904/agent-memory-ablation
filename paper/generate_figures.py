"""Generate the paper's figures from results/summary.json.

Run:  python -m paper.generate_figures
Figures are written to paper/figures/*.png. Deterministic given the results.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(HERE, "figures")

ARMS = ["rag", "graph", "hybrid"]
COLORS = {"rag": "#4C72B0", "graph": "#55A868", "hybrid": "#C44E52"}
LABEL = {"rag": "Stateless RAG", "graph": "Graph-only", "hybrid": "Hybrid"}


def _load():
    with open(os.path.join(ROOT, "results", "summary.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _bars(ax, groups, series, ylabel, title, pct=True, fmt="{:.0f}"):
    import numpy as np

    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for i, (name, vals) in enumerate(series.items()):
        off = (i - (len(series) - 1) / 2) * w
        vv = [v * 100 if pct else v for v in vals]
        bars = ax.bar(x + off, vv, w, label=LABEL.get(name, name),
                      color=COLORS.get(name, None))
        for b, v in zip(bars, vv):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)


def fig_architecture():
    """Schematic of the study: corpus + query sets -> three arms behind one
    interface -> harness metrics + traces. Parity with the originals' Figure 1."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=8.5, tc="black"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                           linewidth=1.1, edgecolor="#333333", facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=13, linewidth=1.1, color="#444444"))

    # inputs
    box(0.3, 8.4, 4.4, 1.2, "Frozen synthetic corpus\n1,583 facts + ground-truth graph\n(closed 22-relation vocabulary)", "#DCE6F1")
    box(5.3, 8.4, 4.4, 1.2, "Frozen query sets (seeded)\n150 known-item / 30 multi-hop\n20 distractor", "#DCE6F1")

    # interface band
    box(0.3, 6.9, 9.4, 0.8, "one typed MemoryProvider interface, build(facts) / recall(query, k) -> refused, ranked facts", "#F2F2F2", fs=8.5)

    # arms
    box(0.3, 4.9, 3.0, 1.5, "Stateless RAG\nMiniLM embeddings\nin sqlite-vec\ntop-k only", "#D8E8DE")
    box(3.5, 4.9, 3.0, 1.5, "Graph-only\ncanonical graph\nforward traversal\nno vectors", "#D8E8DE")
    box(6.7, 4.9, 3.0, 1.5, "Hybrid\nfacts + vectors + graph\nweighted RRF fusion", "#D8E8DE")

    # harness
    box(0.3, 2.7, 6.2, 1.4, "Harness: runner -> metrics\nRecall@k, MRR, multi-hop success,\nrefusal correctness, p50/p95 latency, bytes/fact", "#FBE8D8")
    box(6.9, 2.7, 2.8, 1.4, "OTLP traces\nexported per recall\n(traces/)", "#FBE8D8")

    # outputs
    box(0.3, 1.0, 9.4, 1.0, "results/*.csv  +  results/tables.md (Tables 1-4)  +  figures  ->  paper", "#EFEFEF", fs=9)

    # arrows
    arrow(2.5, 8.4, 2.5, 7.7)
    arrow(7.5, 8.4, 7.5, 7.7)
    arrow(1.8, 6.9, 1.8, 6.4)
    arrow(5.0, 6.9, 5.0, 6.4)
    arrow(8.2, 6.9, 8.2, 6.4)
    arrow(3.4, 4.9, 3.4, 4.1)
    arrow(8.0, 4.9, 8.0, 4.1)
    arrow(3.4, 2.7, 3.4, 2.0)

    ax.set_title("Study design: three arms, one interface, one frozen harness",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig0_architecture.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def fig_known_item(summary):
    cutoffs = ["recall@1", "recall@5", "recall@20"]
    series = {a: [summary[a][c] for c in cutoffs] for a in ARMS}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    _bars(ax, ["R@1", "R@5", "R@20"], series, "Recall (%)",
          "Known-item retrieval (150 queries)")
    ax.set_ylim(0, 108)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_known_item.png"), dpi=150)
    plt.close(fig)


def fig_multihop(summary):
    series = {a: [summary[a]["multihop_success"], summary[a]["multihop_chain_coverage"]]
              for a in ARMS}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    _bars(ax, ["Terminal-fact\nsuccess", "Chain\ncoverage"], series, "Rate (%)",
          "Multi-hop retrieval (30 queries)")
    ax.set_ylim(0, 108)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_multihop.png"), dpi=150)
    plt.close(fig)


def fig_refusal(summary):
    fig, ax = plt.subplots(figsize=(6.0, 4))
    import numpy as np
    x = np.arange(len(ARMS))
    vals = [summary[a]["refusal_correct_distractor"] * 100 for a in ARMS]
    bars = ax.bar(x, vals, 0.6, color=[COLORS[a] for a in ARMS])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}%", ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[a] for a in ARMS])
    ax.set_ylabel("Refusal correctness (%)")
    ax.set_title("Fabrication pressure: refusal on 20 distractors\n(higher = less fabrication)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_refusal.png"), dpi=150)
    plt.close(fig)


def fig_cost(summary, storage):
    """Quality (MRR) vs the two costs: latency and storage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for a in ARMS:
        ax1.scatter(summary[a]["latency_p50_ms"], summary[a]["mrr"] * 100,
                    s=140, color=COLORS[a], label=LABEL[a])
        ax1.annotate(LABEL[a], (summary[a]["latency_p50_ms"], summary[a]["mrr"] * 100),
                     textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax1.set_xlabel("Recall latency p50 (ms)")
    ax1.set_ylabel("Known-item MRR (×100)")
    ax1.set_title("Ranking quality vs latency")
    ax1.grid(alpha=0.25)

    for a in ARMS:
        ax2.scatter(storage[a]["bytes_per_fact"], summary[a]["mrr"] * 100,
                    s=140, color=COLORS[a])
        ax2.annotate(LABEL[a], (storage[a]["bytes_per_fact"], summary[a]["mrr"] * 100),
                     textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax2.set_xlabel("Storage bytes / fact")
    ax2.set_ylabel("Known-item MRR (×100)")
    ax2.set_title("Ranking quality vs storage")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_cost.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    data = _load()
    summary, storage = data["summary"], data["storage"]
    fig_architecture()
    fig_known_item(summary)
    fig_multihop(summary)
    fig_refusal(summary)
    fig_cost(summary, storage)
    print(f"wrote figures to {FIGDIR}")


if __name__ == "__main__":
    main()
