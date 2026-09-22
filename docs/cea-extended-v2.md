# CEA-Extended: dataset schema v2

Schema v2 is an experimental extension of the published CEA v0.1 protocol. It does not change the archived paper pilot, schema v1, or the `paper-v0.1` scoring profile. The package development version is 0.3.0; the 0.2.0 release and tag remain the v1 reference.

## Research question and intervention matrix

For each case, does the answer follow decisive evidence changes, remain stable under irrelevant changes, abstain when every proof path is broken, continue answering when an alternative proof path survives, and apply a declared conflict policy? Every v2 case must contain exactly one `baseline`, at least two `counterfactual` conditions, and at least one each of `nuisance`, `ablation`, `redundancy`, and `conflict`. A missing family is a validation error, so a high joint score cannot hide an omitted extension.

Each condition contains a full set of documents with stable IDs, an `expected_answer` or `null` for abstention, and all admissible **minimal** proof sets. A proof set is a collection of document IDs that jointly supports the answer. More than one proof set permits alternate valid citation paths. The sets must form an antichain: no set may contain another. The runner accepts a citation set for coverage when it contains **any** admissible proof set; strict scoring requires equality with **one** admissible proof set. Both also require a correct answer, valid available citation IDs, a valid response format, and no target error. For abstention, coverage permits citations to available documents, while strict requires no citations.

## JSON shape

See the executable [`cea-extended-demo.json`](../datasets/cea-extended-demo.json) and the [v2 JSON Schema](../schema/audit-dataset-v2.schema.json). A condition has this shape:

```json
{
  "id": "baseline",
  "family": "baseline",
  "documents": [{"id": "D1", "text": "A claim", "metadata": {}}],
  "expected_answer": "An answer",
  "answer_aliases": [],
  "proof_sets": [["D1"]],
  "changed_document_ids": [],
  "removed_document_ids": [],
  "rationale": "A reviewer explains why the claim entails the answer."
}
```

`conditions` is an array on each case. `resolution_policy` is required only on a `conflict` condition: `authority` requires a non-null answer and `authority_document_ids` included in a proof; `abstain` requires a null answer and no authority IDs. Each condition describes the actual edit against the baseline with `changed_document_ids` and `removed_document_ids`. The validator disallows new IDs and checks that those declarations equal the document diff. Order permutations need no changed ID. Every non-baseline condition must have a non-empty human-review `rationale`.

The validator additionally checks that counterfactual answers differ from baseline and their evidence worlds differ; nuisance preserves the baseline answer and proof family; ablation removes at least one member of **every** baseline proof set; redundancy removes one proof path while retaining at least one, with all surviving paths declared. Independent counterfactuals may lead to the same alternative answer. A conflict's factual and authority coherence, like the entailment and minimality of all declared proofs, **cannot be proven by these structural checks**. Reviewers must inspect the source meaning, all alternative proof paths, and whether the removal genuinely leaves no answer.

## Run and interpret

```bash
causal-rag-audit validate datasets/cea-extended-demo.json
causal-rag-audit run \
  --dataset datasets/cea-extended-demo.json \
  --target my_package.audit_adapter:target \
  --scoring-profile strict-exact \
  --output extended-results \
  --minimum extended_strict_causal_evidence_score=1.0
```

Use `strict-exact` for arbitrary stable citation IDs; the default `paper-v0.1` parser only accepts uppercase `D<number>` IDs. The report records every declared condition, per-family counts and pass rates, all admissible proof sets, which proof set matched, answer/citation/format diagnostics, and proof recall and precision for answerable conditions. `extended_coverage_causal_evidence_score` and `extended_strict_causal_evidence_score` are hard conjunctions across every condition in a case. The Wilson 95% intervals are descriptive **per-case** intervals under independent-case sampling. They do not account for repeated templates, shared domains, or clustered source corpora; choose the actual independent sampling unit before making population claims.

This is an audit of the answerer when it receives controlled documents. Retrieval over a changed corpus requires a separate isolated-index audit. A passing v2 case shows behavioral consistency under its declared edits, not universal evidence use, source truth, or internal model causality.

## Migration from v1

Existing `schema_version: 1` files and Python calls work as before. Preserve v1 datasets to reproduce the 384 recorded pilot generations and their `coverage_causal_evidence_score` / `strict_causal_evidence_score`. A v2 case needs explicit `conditions` and `proof_sets`; migration requires authoring and reviewing the new interventions, not an automatic field rename. V2 produces separate `extended_*` joint metrics and a schema-2 report. A release gate should name the intended metric and scoring profile explicitly.
