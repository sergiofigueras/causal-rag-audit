from __future__ import annotations

import unittest
from pathlib import Path

from causal_rag_audit import (
    PAPER_PROFILE,
    STRICT_PROFILE,
    TargetResponse,
    audit,
    load_dataset,
)
from causal_rag_audit.runner import coerce_response
from examples.demo_rag import grounded_rag

ROOT = Path(__file__).resolve().parents[1]
DATASET = load_dataset(ROOT / "examples" / "audit_cases.json")


class PaperProfileTests(unittest.TestCase):
    def test_wrapper_answer_and_long_abstention_match_paper_rules(self) -> None:
        def target(request):
            if "coolant" not in request.question.casefold():
                return grounded_rag(request)
            texts = " ".join(document.text for document in request.documents)
            if "Argon-X" in texts:
                return '{"answer":"The Argon-X coolant","citations":["D1"]}'
            if "Neon-Y" in texts:
                return '{"answer":"Neon-Y","citations":["D1"]}'
            return (
                '{"answer":"The coolant is not specified in the provided sources",'
                '"citations":[]}'
            )

        report = audit(target, DATASET, scoring_profile=PAPER_PROFILE)
        direct = report.cases[0]
        self.assertTrue(direct.indicators["paired_responsiveness"])
        self.assertTrue(direct.indicators["ablation_abstention"])
        self.assertTrue(direct.indicators["strict_causal_evidence_score"])

    def test_strict_exact_profile_rejects_unlisted_wrappers(self) -> None:
        def target(request):
            response = grounded_rag(request)
            if response.answer == "Argon-X":
                return TargetResponse("The Argon-X coolant", response.citations)
            return response

        report = audit(target, DATASET, scoring_profile=STRICT_PROFILE)
        self.assertFalse(report.cases[0].responses[0].correct)

    def test_malformed_citations_do_not_erase_a_parseable_answer(self) -> None:
        def target(request):
            response = grounded_rag(request)
            if response.answer == "Argon-X":
                return '{"answer":"Argon-X","citations":[1]}'
            return response

        report = audit(target, DATASET, scoring_profile=PAPER_PROFILE)
        response = report.cases[0].responses[0]
        self.assertEqual(response.answer, "Argon-X")
        self.assertTrue(response.correct)
        self.assertFalse(response.format_valid)
        self.assertEqual(response.citations, ())
        self.assertIn("ResponseFormatError", response.error)

    def test_paper_schema_marks_extra_fields_invalid_but_keeps_scores(self) -> None:
        response = coerce_response(
            {"answer": "Argon-X", "citations": ["D1"], "trace": "hidden"},
            scoring_profile=PAPER_PROFILE,
        )
        self.assertEqual(response.answer, "Argon-X")
        self.assertEqual(response.citations, ("D1",))
        self.assertFalse(response.format_valid)

    def test_strict_profile_deduplicates_but_marks_format_invalid(self) -> None:
        response = coerce_response(
            {"answer": "Argon-X", "citations": ["D1", "D1"]},
            scoring_profile=STRICT_PROFILE,
        )
        self.assertEqual(response.citations, ("D1",))
        self.assertFalse(response.format_valid)

    def test_custom_judges_override_profile_matching(self) -> None:
        def answer_judge(actual, expected, _aliases, _alternatives):
            return actual.casefold().endswith(expected.casefold())

        def abstention_judge(actual, _accepted):
            return actual == "NO-EVIDENCE"

        def target(request):
            response = grounded_rag(request)
            if response.answer == "INSUFFICIENT":
                return TargetResponse("NO-EVIDENCE", ())
            return TargetResponse(f"answer: {response.answer}", response.citations)

        report = audit(
            target,
            DATASET,
            scoring_profile=STRICT_PROFILE,
            answer_judge=answer_judge,
            abstention_judge=abstention_judge,
        )
        self.assertEqual(report.metrics["strict_causal_evidence_score"].rate, 1.0)


if __name__ == "__main__":
    unittest.main()
