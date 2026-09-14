# Causal RAG Audit

[![CI](https://github.com/sergiofigueras/causal-rag-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/sergiofigueras/causal-rag-audit/actions/workflows/ci.yml)
[![Python 3.10–3.14](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`causal-rag-audit` tests whether a RAG answerer *behaviorally relies* on the evidence it receives. It runs the same question in three controlled conditions:

1. **Original world:** the answer is supported by an annotated proof.
2. **Counterfactual world:** a minimal evidence change makes a different answer correct.
3. **Ablated world:** necessary evidence is removed, so the system should abstain.

This catches a failure that ordinary groundedness evaluation can miss: a system may return the expected answer and cite a compatible document without changing its answer when that evidence changes or disappears.

The framework wraps an existing RAG answerer. It does **not** build a retriever, call a particular model vendor, or claim to identify causes inside a neural network.

> **Status:** v0.1 is an alpha research implementation. Its API and dataset schema are versioned, but may evolve with empirical use.

## Quick start

Requirements: Python 3.10 or newer. The framework itself has no runtime dependencies.

```bash
git clone https://github.com/sergiofigueras/causal-rag-audit.git
cd causal-rag-audit
python -m pip install .

causal-rag-audit validate examples/audit_cases.json
causal-rag-audit run \
  --dataset examples/audit_cases.json \
  --target examples.demo_rag:grounded_rag \
  --output audit-results \
  --minimum strict_causal_evidence_score=1.0
```

The run makes six calls—three conditions for each of two cases—and writes:

- `audit-results/report.json`: complete machine-readable evidence;
- `audit-results/report.md`: metrics and per-case failure diagnostics.

Now run the deliberately broken shortcut target:

```bash
causal-rag-audit run \
  --dataset examples/audit_cases.json \
  --target examples.demo_rag:shortcut_rag \
  --output shortcut-results
```

It gives plausible original-world answers and citations but ignores changed and missing evidence. Accordingly, observational support passes while the causal scores fail.

## Wrap your RAG system

### Python callable

A target accepts a `TargetRequest` and returns a `TargetResponse` or an equivalent dictionary. The request intentionally contains no oracle answer and no condition label.

```python
from causal_rag_audit import TargetRequest, TargetResponse, audit


def my_rag(request: TargetRequest) -> TargetResponse:
    result = existing_rag_answer(
        question=request.question,
        documents=[document.to_dict() for document in request.documents],
    )
    return TargetResponse(
        answer=result.answer,
        citations=tuple(result.document_ids),
    )


report = audit(
    my_rag,
    "my_audit_cases.json",
    target_name="customer-support-rag-v3",
    max_workers=1,
)
print(report.metrics["strict_causal_evidence_score"].rate)
```

Your target must return exactly:

```json
{
  "answer": "the answer or a declared abstention token",
  "citations": ["stable-document-id"]
}
```

Use `KeywordTarget` if your function already accepts `question`, `documents`, and `metadata` keyword arguments. See [integration recipes](docs/integrations.md) for LangChain-, LlamaIndex-, and Haystack-shaped wrappers.

### HTTP endpoint

To audit a service without importing its code:

```bash
export RAG_TOKEN='replace-me'
causal-rag-audit run \
  --dataset my_audit_cases.json \
  --http-url https://rag.example.test/audit-answer \
  --header-from-env 'Authorization=RAG_TOKEN' \
  --timeout 90 \
  --output audit-results
```

The framework sends this JSON by `POST`:

```json
{
  "question": "Which coolant does Reactor Vega use?",
  "documents": [
    {"id": "D1", "text": "...", "metadata": {}}
  ],
  "metadata": {}
}
```

The endpoint returns the same `answer`/`citations` object shown above. Secrets are read from environment variables and are not added to reports.

## Author an audit dataset

Start from [`examples/audit_cases.json`](examples/audit_cases.json) and validate every change:

```bash
causal-rag-audit validate my_audit_cases.json
```

Each case declares:

- the same stable document IDs in `world0` and `world1`;
- different oracle answers for the two worlds;
- minimal proof document IDs for each answer;
- the document text/metadata changes, split into `causal_change_ids` and optional `nuisance_change_ids`;
- world-0 proof documents whose removal makes the question unanswerable;
- optional exact-match answer aliases and case metadata.

Validation goes beyond the included [JSON Schema](schema/audit-dataset-v1.schema.json): it checks cross-world ID equality, declarations against actual changes, proof references, ablation validity, disjoint causal/nuisance changes, and answer separation.

Good audit cases require domain expertise. A schema-valid case can still be scientifically invalid if another retained document supports the answer or the two worlds differ in an uncontrolled way. Follow the [authoring protocol](docs/authoring-audits.md) and have a second reviewer inspect the cases.

## Metrics

Every metric is a hard per-case indicator, macro-averaged over cases with a Wilson 95% interval.

| Metric | Pass condition |
|---|---|
| `world0_accuracy` / `world1_accuracy` | Exact normalized answer or a predeclared alias matches that world. |
| `paired_responsiveness` (CRC) | Both world answers are correct and the system changes its answer. |
| `ablation_abstention` (ENA) | The ablated request returns a declared abstention. |
| `ablation_citation_validity` | Every ablation citation names a document that remains available. |
| `observational_support` (OBS) | The original answer is correct and its citations cover the proof. |
| `coverage_causal_evidence_score` (CES) | CRC + ENA + valid ablation citations + proof coverage in both full worlds. |
| `strict_causal_evidence_score` (CES-strict) | CRC + ENA + no ablation citations + exact proof citations in both full worlds. |
| `format_validity` | All three responses satisfy the response contract. |

Coverage CES allows extra valid citations in full worlds; strict CES rejects them. Both are intentionally conjunctive: a case passes only when all required behaviors occur together. Read the [concepts and metric rationale](docs/concepts.md) before interpreting results.

## Where it works

The current framework is a good fit when:

- you can supply controlled evidence directly to an answer-generation boundary;
- documents have stable IDs and the answerer emits document-level citations;
- questions have reviewable, short oracle answers and explicit evidence proofs;
- counterfactual evidence can be made coherent without changing unrelated difficulty;
- removing declared evidence genuinely makes the question unanswerable;
- you want an offline regression, release gate, model/prompt comparison, or red-team audit.

It can wrap local models, hosted models, deterministic pipelines, and vendor-neutral HTTP services. CI exercises the package on Linux, macOS, and Windows with supported Python versions; your RAG stack may have additional constraints.

## Where it does not work

Do not use v0.1 as evidence for claims it does not test:

- **No controlled context:** it cannot intervene on a service that chooses hidden context and exposes no injection boundary.
- **No stable citations:** it does not infer evidence provenance from free-form prose.
- **Open-ended judgment:** ambiguous, creative, opinion, or many-valid-answer tasks do not fit conservative exact scoring without carefully declared aliases.
- **Source truth or quality:** behavioral sensitivity to a document does not make that document true, current, safe, or authoritative.
- **Internal model causality:** the audit observes input-output behavior; it does not locate neural mechanisms or prove model intent.
- **Retriever quality:** the default protocol audits the answerer over supplied documents. End-to-end retrieval requires your adapter to build/query an isolated index for every world; v0.1 does not manage that lifecycle.
- **Production traffic guarantees:** synthetic audit performance does not estimate all live queries, populations, or distribution shifts.
- **Claim-level/multimodal evidence:** v0.1 scores short answers and document IDs, not spans, images, audio, tables, or multi-claim citation alignment.
- **Async callable targets:** direct callables must be synchronous. Put async applications behind a synchronous adapter or HTTP endpoint.

More detail is in [limitations and valid claims](docs/limitations.md).

## Operational guidance

- Use `--workers 1` unless the target is thread-safe and its provider permits concurrency.
- Target exceptions become failed, inspectable response rows; a run is not silently retried.
- Add repeated runs yourself for stochastic systems and report the distribution. One run is not a determinism claim.
- Raw target payloads are excluded by default. `--include-raw` may store prompts, model traces, personal data, or vendor metadata in `report.json`; review before sharing.
- A CLI run exits `2` if target calls fail or a `--minimum METRIC=RATE` gate is missed, and `1` for configuration/dataset errors.

## Development

```bash
python -m pip install .
make check
```

`make check` compiles the source, runs the standard-library test suite, validates the example dataset, and executes the passing demo with a strict CES gate. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## Research relationship

This software operationalizes the protocol introduced in **“Beyond Citation Entailment: Causal Evidence Audits for Retrieval-Augmented Language Models”**. The paper, benchmark-generation materials, raw generations, and research reproduction artifacts remain in [sergiofigueras/causal-evidence-audit](https://github.com/sergiofigueras/causal-evidence-audit). This repository is the reusable integration framework for auditing other RAG systems.

When using the software academically, cite both records described in [`CITATION.cff`](CITATION.cff).

## License

Code and the synthetic demo dataset are released under the [MIT License](LICENSE).
