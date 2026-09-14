# Concepts and metric rationale

## The gap: compatible evidence is not necessarily used evidence

A conventional groundedness check observes one evidence world. If the answer is correct and a cited passage entails it, the result looks good. But the same observation is also compatible with a shortcut: the answerer may rely on memorization, a fixed prior, question cues, or some undisclosed source and then attach a plausible citation.

Causal RAG Audit adds controlled input interventions. For a fixed question:

- `world0` supports answer `A0` with proof `P0`;
- `world1` changes declared evidence and supports a distinct answer `A1` with proof `P1`;
- `ablated` removes necessary documents from `world0`, so the valid behavior is abstention.

The target sees the question, documents, and user metadata. It never receives the condition name, oracle answer, proof, or change declarations. Therefore, success requires the answer to track the supplied evidence rather than an evaluator hint.

## Per-case indicators

Let `correct(w)` mean the output matches the oracle under the preregistered scoring profile or custom judge in world `w`. Let `covers(w)` mean its cited ID set includes the annotated proof, `exact(w)` mean the cited set equals that proof, `abstains(a)` mean an allowed abstention is returned after ablation, and `valid(a)` mean all ablation citations still exist.

For each full world, proof citation recall is `|C ∩ G| / |G|` and proof citation precision is `|C ∩ G| / |C|`, with precision zero for empty citations when the proof is non-empty. Reports macro-average both quantities separately for world 0 and world 1.

- **CRC / paired responsiveness:** `correct(0) AND correct(1) AND answer(0) != answer(1)`.
- **ENA / ablation abstention:** `abstains(a)`.
- **OBS / observational support:** `correct(0) AND covers(0)`.
- **Coverage CES:** `CRC AND ENA AND valid(a) AND covers(0) AND covers(1)`.
- **Strict CES:** `CRC AND ENA AND no_citations(a) AND exact(0) AND exact(1)`.

The aggregate is the proportion of cases passing each hard indicator. The report also gives a Wilson 95% binomial interval. That interval reflects finite case count; it does not correct biased case construction, model stochasticity, or domain shift.

## Why two CES variants?

Some production systems deliberately include contextual citations beyond a minimal proof. Coverage CES permits this when all evidence-sensitive behavior is correct. Strict CES is useful when citation precision itself matters: it requires exactly the annotated proof and no citation after abstention.

Neither score should automatically be treated as a universal quality metric. Choose and preregister the variant that matches the system contract.

## Unit of intervention

Version 0.2 intervenes on a caller-supplied set of documents. This clean boundary lets the same evaluator wrap many generators, but it means default results describe **generation conditional on controlled context**, not the quality of a production retriever.

An end-to-end experiment is possible only when the target adapter can create an isolated corpus/index for each call, retrieve from it, and return stable source IDs. That lifecycle belongs to the integration in v0.2 and must not leak state between worlds.

## Valid conclusion

A high score supports a bounded statement:

> Under these audited questions, declared interventions, target configuration, and sampled runs, the observable answers and citations followed the supplied evidence according to the selected scoring contract.

It does not prove source truth, universal robustness, human-like reasoning, model intent, or neural-level causality.
