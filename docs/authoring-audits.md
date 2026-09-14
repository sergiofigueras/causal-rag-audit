# Authoring defensible audit cases

The code can enforce structural invariants, but scientific validity depends on case design. Treat dataset creation as an annotation protocol, not a text-generation convenience.

## Recommended workflow

1. **Fix the question and task contract.** Prefer one short, objectively scorable answer. State what counts as abstention before running targets.
2. **Construct world 0.** Include all and only the context that the target would legitimately receive. Mark a minimal set of document IDs sufficient for the answer.
3. **Construct world 1.** Keep the question and ordered document IDs fixed. Change the smallest evidence unit that coherently makes a distinct answer correct. Declare every changed document as causal or nuisance. Test order permutations separately rather than mixing them into the answer-changing pair.
4. **Re-annotate the proof.** Do not assume the world-0 proof remains valid; record a minimal proof for each world independently.
5. **Construct the ablation.** Remove one or more world-0 proof documents such that no retained document, metadata field, or simple combination still reveals the answer.
6. **Add nuisance changes only for a reason.** A nuisance change may control superficial cues, but it cannot belong to either proof. Balance nuisance patterns across the dataset.
7. **Blind review.** Ask a second reviewer to answer all three conditions from the presented documents alone and to challenge proof minimality.
8. **Validate and pilot.** Run `causal-rag-audit validate`, then inspect actual failures rather than tuning aliases after seeing a preferred model's output.

## Minimal example

```json
{
  "id": "coolant-001",
  "question": "Which coolant does Reactor Vega use?",
  "world0": [
    {"id": "D1", "text": "Reactor Vega uses Argon-X.", "metadata": {}},
    {"id": "D2", "text": "Reactor Vega is inspected Monday.", "metadata": {}}
  ],
  "world1": [
    {"id": "D1", "text": "Reactor Vega uses Neon-Y.", "metadata": {}},
    {"id": "D2", "text": "Reactor Vega is inspected Monday.", "metadata": {}}
  ],
  "answers": {"world0": "Argon-X", "world1": "Neon-Y"},
  "proofs": {"world0": ["D1"], "world1": ["D1"]},
  "ablate_document_ids": ["D1"],
  "causal_change_ids": ["D1"],
  "nuisance_change_ids": [],
  "answer_aliases": {"world0": [], "world1": []},
  "metadata": {"domain": "synthetic"}
}
```

## Common invalid cases

- World 1 changes tone, length, or entity rarity at the same time as the answer-bearing fact, creating unintended cues.
- The answer changes only in metadata that the target never reads.
- An ablated world retains a duplicate, title, filename, table cell, or neighboring fact that still identifies the answer.
- A “minimal proof” omits a bridge document needed for multi-hop reasoning.
- Aliases are added from observed model mistakes rather than independent annotation.
- Real records are edited into counterfactuals that are internally inconsistent or harmful if mistaken for truth.
- Repeated templates make the answer position predictable.

## Dataset reporting checklist

Publish the domain, case count, sampling procedure, authoring procedure, number and independence of reviewers, disagreement resolution, proof-length distribution, intervention types, nuisance controls, abstention contract, scoring variant, target settings, repeat count, and exclusions. Preserve the dataset fingerprint emitted in every report.

Synthetic examples are ideal for debugging the protocol. Claims about a real domain require representative domain cases and expert review.
