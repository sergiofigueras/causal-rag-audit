# Integration recipes

The stable boundary is intentionally small: controlled documents go in; one answer and stable document IDs come out. The snippets below show adapter shapes, not endorsements of a particular RAG library.

Choose `paper-v0.1` for the manuscript's `D<number>` response convention or `strict-exact` for arbitrary stable IDs. If you use UUIDs, URLs, or framework-native node IDs, select `strict-exact` explicitly.

## Existing keyword function

```python
from causal_rag_audit import KeywordTarget, audit


def answer(*, question, documents, metadata):
    result = app.answer(question=question, context=documents)
    return {"answer": result.text, "citations": result.source_ids}


report = audit(KeywordTarget(answer), "audit.json")
```

## LangChain-shaped pipeline

Inject the audit documents as the context instead of invoking the production retriever:

```python
from causal_rag_audit import TargetResponse


def langchain_target(request):
    context = "\n\n".join(f"[{d.id}] {d.text}" for d in request.documents)
    result = chain.invoke({"question": request.question, "context": context})
    return TargetResponse(
        answer=result["answer"],
        citations=tuple(result["source_ids"]),
    )
```

The prompt must require machine-readable stable IDs. Do not infer citations by fuzzy matching after generation if your claim concerns the application's own citation behavior.

## LlamaIndex-shaped query engine

For each call, construct nodes from the supplied documents, preserving `document.id` as the node/source ID. Query only that ephemeral index and map the returned source nodes back to those IDs:

```python
def llamaindex_target(request):
    nodes = [make_node(text=d.text, node_id=d.id) for d in request.documents]
    engine = make_isolated_query_engine(nodes)
    result = engine.query(request.question)
    return {
        "answer": str(result),
        "citations": [node.node_id for node in result.source_nodes],
    }
```

If an index is cached between calls, counterfactual documents can contaminate one another and invalidate the audit.

## Haystack-shaped pipeline

Create framework documents with `id=document.id`, pass them through a pipeline branch that accepts caller-provided documents, then return the answer plus document IDs:

```python
def haystack_target(request):
    supplied = [make_document(id=d.id, content=d.text, meta=d.metadata) for d in request.documents]
    result = pipeline.run({"prompt_builder": {"question": request.question, "documents": supplied}})
    answer = result["answer_builder"]["answers"][0]
    return {"answer": answer.data, "citations": [doc.id for doc in answer.documents]}
```

## HTTP service

Expose an audit-only route that accepts the documented request JSON. Keep authentication in an environment variable:

```bash
export AUDIT_AUTH='Bearer ...'
causal-rag-audit run \
  --dataset audit.json \
  --http-url http://localhost:8000/audit-answer \
  --header-from-env 'Authorization=AUDIT_AUTH'
```

An audit-only route is often safer than adding test controls to a public production route. Apply normal authorization, request-size, logging, and data-retention policies.

## CI release gate

```bash
causal-rag-audit run \
  --dataset regression-audit.json \
  --target my_package.audit_adapter:target \
  --output artifacts/causal-audit \
  --scoring-profile strict-exact \
  --minimum paired_responsiveness=0.95 \
  --minimum ablation_abstention=0.95 \
  --minimum coverage_causal_evidence_score=0.90 \
  --minimum world0_proof_citation_precision=0.90
```

Thresholds should be fixed from product risk and a reviewed baseline, not selected after inspecting the candidate run. Archive `report.json`, record model/prompt/index versions through `target_metadata` in the Python API, and repeat stochastic targets enough times to characterize variability.

## Concurrency and state

`max_workers > 1` invokes a target concurrently from threads. Enable it only if the entire target, client library, rate limit, and temporary-index strategy are thread-safe. Result order is deterministic even when execution order is not. The framework does not retry calls, so provider errors remain visible rather than being selectively hidden.
