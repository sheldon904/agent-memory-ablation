# agent-memory-ablation

A **controlled, reproducible ablation** of three agent-memory architectures on one frozen
synthetic corpus and one frozen evaluation harness. Stateless RAG vs. a canonical
knowledge graph vs. their hybrid, head to head, every number reproducible from
`make eval`.

This is the external-baseline experiment that a companion production study
([hermes-hybrid-memory](https://github.com/sheldon904/hermes-hybrid-memory)) named as its
own most important untested claim, generalized into a three-way ablation and pointed back
at the author's own design assumptions. **The full writeup:**
[`paper/README.md`](paper/README.md) (GitHub) ·
[`agent-memory-ablation.pdf`](paper/agent-memory-ablation.pdf) (PDF) ·
[`paper.tex`](paper/paper.tex) (arXiv LaTeX).

## The headline

On a clean, canonical corpus (1,583 facts, entity-anchored queries), the results do *not*
flatter the hybrid:

| Metric | Stateless RAG | Graph-only | Hybrid |
|---|---|---|---|
| Known-item Recall@5 | 68.7% | **98.7%** | 98.0% |
| Known-item MRR | 0.610 | 0.750 | **0.800** |
| Multi-hop success (top-10) | 0.0% | **53.3%** | 40.0% |
| **Refusal on distractors** (↑ = less fabrication) | 10.0% | **100.0%** | 10.0% |
| Latency p50 (host-dependent) | 7.7 ms | **0.30 ms** | 9.0 ms |
| Storage bytes/fact | 2,049 | **101** | 2,528 |

The vector-free **graph arm wins or ties most axes** and *never fabricates*; the
**hybrid's advantage reduces to ranking precision** (Recall@1, MRR), bought at ~30× the
latency, 25× the storage, and a 90-point refusal regression it inherits from its dense
channel. Two of four pre-registered hypotheses are refuted. See
[**§8 "Where the hybrid loses"**](paper/README.md#8-where-the-hybrid-loses) for the honest
accounting, and [§10 Threats](paper/README.md#10-threats-to-validity) for why this corpus
is the graph's best case.

## Quickstart

```bash
pip install -r requirements.txt
make eval          # or, without make:  python scripts/run.py eval
make test          # 25-test suite
```

`make eval` regenerates the frozen corpus and query sets (verifying their committed
SHA-256 checksums), builds all three arms, runs 200 queries, and writes `results/*.csv`,
`results/tables.md` (the exact source of the paper's Tables 1-4), `results/summary.json`,
OTLP traces under `traces/`, and the figures.

**Fully offline / CI:** `AMA_EMBEDDER=hash make eval` runs the entire study with a
deterministic, dependency-free hashing embedder, no model download. Absolute numbers
differ, but because all three arms share the embedder, the architectural comparison holds.

## What makes it reproducible

- **Frozen corpus** (`corpus/`, seed 42): a deterministic generator builds a fictional
  field-services company with a fully known ground-truth relation graph. Committed with
  checksums; the test suite fails if the committed artifact drifts from its seed.
- **Frozen query sets** (`harness/queries/`, seed 1234): 150 known-item, 30 multi-hop
  (2-3 joins), 20 distractor (answer absent), built *before* any arm was run, ground
  truth computed by graph traversal.
- **Pinned dependencies** (`requirements.txt`): real `all-MiniLM-L6-v2` embeddings in
  `sqlite-vec`; deterministic to float32.
- **One typed interface** (`arms/base.py`): every arm is a `MemoryProvider`; the harness
  derives every metric from `recall()` output, so no arm sees ground truth.

Only latency is host-dependent (and reported as measured); all retrieval metrics
reproduce bit-for-bit on the same platform + model version.

## Repository layout

```
corpus/     frozen synthetic corpus + deterministic generator (facts, graph, checksums)
arms/       three memory arms behind one MemoryProvider interface
              base.py  embeddings.py  entities.py  vector_index.py
              rag.py   graph.py       hybrid.py
harness/    seeded query generator, runner, metrics, OTel tracing
              build_queries.py  run_eval.py  metrics.py  tracing.py  queries/
results/    generated CSVs + auto-generated tables.md + summary.json
traces/     exported OTLP spans, one per recall (see traces/README.md)
paper/      the writeup, three ways: README.md (renders on GitHub),
              agent-memory-ablation.pdf (rendered PDF, `make pdf`), paper.tex (arXiv LaTeX),
              plus the figure + PDF generators and figures/
tests/      corpus / query / arm / metric integrity + determinism (make test)
scripts/    run.py, cross-platform, make-free entrypoint
```

## The three arms

- **Stateless RAG**, chunked text, MiniLM embeddings in `sqlite-vec`, exact-cosine top-k,
  a single cosine floor for both relevance and refusal. No graph, no state.
- **Graph-only**, the canonical-graph, closed-vocabulary posture: entity-linked forward
  traversal over the ground-truth relation graph, ranked by hop distance + lexical
  overlap. No vectors. Refuses when the query links no entity.
- **Hybrid**, a ported slice of a five-channel design: facts (BM25) + vectors (MiniLM) +
  graph, fused with weighted Reciprocal Rank Fusion. The minimal realization of the
  rank-fusion improvement the production study recommended for itself.

## Related work (same author, same house style)

1. *Augmented Operational Decisioning with Ontology-Grounded Local Agents*, Little Bear
   Foundry, May 2026.
2. *Hybrid Holographic Memory in a Production Personal Agent*, 
   [hermes-hybrid-memory](https://github.com/sheldon904/hermes-hybrid-memory), July 2026.
   This repository runs the external-baseline experiment that study named as its top open
   item.

## License

MIT, see [LICENSE](LICENSE).
