# Limitations and valid claims

## What the result measures

The audit measures observable response sensitivity under a declared set of document interventions. It combines answer correctness, evidence change, ablation abstention, citation coverage/precision, and citation validity at document granularity.

It is strongest as a paired regression test: the question and most context remain fixed, while the answer-bearing evidence changes or disappears.

## What the result does not establish

- that any evidence document is factually true, safe, unbiased, licensed, or current;
- that retrieval ranks or selects the best evidence unless an isolated retrieval lifecycle is explicitly part of the adapter;
- that the model internally reasoned over, attended to, or intended to use a citation;
- that the system will behave similarly outside the sampled domains and templates;
- that an audit dataset is valid merely because it passes structural validation;
- that one successful execution is deterministic or statistically representative;
- claim-level support when an answer contains several independently verifiable claims;
- span-, sentence-, table-, image-, audio-, or video-level provenance;
- calibrated uncertainty beyond exact, predeclared abstention forms;
- protection from target code that recognizes or has memorized a public benchmark.

## Current engineering boundaries

- Python 3.10+ and synchronous target callables; HTTP services may implement their internals asynchronously.
- JSON datasets and JSON/Markdown reports.
- Short-answer exact matching after Unicode, case, punctuation, and whitespace normalization. Synonyms must be independently declared as aliases.
- Document-level citations identified by exact stable strings.
- One ablation derived from world 0 per case.
- No built-in model provider, prompt, retriever, vector store, judge model, randomization scheduler, repeated-run estimator, or experiment tracker.
- Thread concurrency only; no distributed runner.

## Threats to validity

**Construct validity.** CES operationalizes one specific notion of evidence reliance. Other useful behaviors—such as synthesizing conflicting evidence or appropriately trusting a prior—need different audits.

**Case leakage.** A target may have memorized public examples or identify repeated templates. Use private held-out cases, vary surface forms without changing difficulty, and never send evaluator-only fields to the target.

**Intervention artifacts.** Counterfactual edits can introduce fluency, length, rarity, or consistency cues. Human review and balanced nuisance controls reduce but do not eliminate this risk.

**Ablation validity.** If retained evidence still implies the answer, abstention is not the right oracle. Review the entire context rather than only the declared proof.

**Stochasticity.** Rates from one sample of generations can move on rerun. Repeat the complete paired audit, retain all reports, and describe aggregation before making comparative claims.

**External validity.** A score over synthetic or narrow cases cannot be generalized to all production requests without a defensible sampling design.

## Responsible reporting language

Prefer:

> The target passed coverage CES on X of N reviewed cases in this dataset under the recorded configuration.

Avoid:

> The model truly used its sources, is causally faithful, or cannot hallucinate.

The first claim is auditable and bounded. The second exceeds what an input-output intervention can establish.
