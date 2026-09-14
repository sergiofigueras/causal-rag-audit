# Scoring profiles and custom judges

Scoring behavior must be fixed before evaluating a target. Every report records the selected profile and the names of custom judge callables.

## `paper-v0.1`

This default profile reproduces the parser and matching rules used for the paper's CEval-32 pilot:

- case and punctuation-insensitive alphanumeric answer normalization;
- the expected fictional candidate may appear in a wrapper phrase, but the alternative candidate must not appear;
- any predeclared abstention marker may occur within a longer response;
- citations use the pilot's `D<number>` IDs, are uppercased, filtered, sorted, and treated as a set;
- the response JSON must contain exactly `answer` and `citations`;
- a malformed citation field fails format validity without erasing a parseable answer.

Use this profile to reproduce the released pilot or to follow the same output contract. It is inappropriate for citation IDs that do not follow `D<number>`.

## `strict-exact`

This profile supports arbitrary non-empty string document IDs. Answers and abstentions must equal the oracle, an alias, or a declared abstention after Unicode, case, punctuation, and whitespace normalization. The response may contain `answer`, `citations`, and optional `metadata`; unknown fields fail format validity. Duplicate citations are deduplicated for set-based scoring but fail the strict format contract.

```python
report = audit(
    target,
    "audit.json",
    scoring_profile="strict-exact",
)
```

## Custom Python judges

When neither built-in rule is suitable, pass fixed judge callables. An answer judge receives `(actual, expected, aliases, alternatives)`. An abstention judge receives `(actual, accepted_abstentions)`.

```python
def answer_judge(actual, expected, aliases, alternatives):
    accepted = {expected.casefold(), *(alias.casefold() for alias in aliases)}
    return actual.casefold() in accepted


def abstention_judge(actual, accepted_abstentions):
    return actual.casefold().startswith("cannot answer:")


report = audit(
    target,
    "audit.json",
    scoring_profile="strict-exact",
    answer_judge=answer_judge,
    abstention_judge=abstention_judge,
)
```

Custom judges are available through the Python API because a CLI-imported judge would execute arbitrary local code without an explicit integration wrapper. Put a project-specific judge next to the target adapter, test it independently, preregister its behavior, and preserve its source revision with the report.

## Format validity is independent

A response can have a scoreable answer and invalid formatting. The framework retains the parsed answer and valid citation IDs, records `format_valid=false`, attaches a `ResponseFormatError` issue, and continues scoring. CLI runs still exit `2` when any response issue occurs.

This separation is deliberate: malformed citation syntax must not retroactively turn an otherwise correct answer into an answer-accuracy failure. Consumers that require perfect structure should gate on `format_validity=1` as well as the causal metrics.
