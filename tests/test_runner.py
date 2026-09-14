from __future__ import annotations

import time
import unittest
from pathlib import Path

from causal_rag_audit import AuditRunner, TargetResponse, audit, load_dataset
from causal_rag_audit.runner import (
    ResponseFormatError,
    coerce_response,
    dataset_fingerprint,
)
from examples.demo_rag import grounded_rag, shortcut_rag

ROOT = Path(__file__).resolve().parents[1]
DATASET = load_dataset(ROOT / "examples" / "audit_cases.json")


class RunnerTests(unittest.TestCase):
    def test_grounded_target_passes_every_metric(self) -> None:
        report = audit(grounded_rag, DATASET, target_name="grounded-demo")
        self.assertEqual(report.error_count, 0)
        self.assertTrue(all(metric.rate == 1.0 for metric in report.metrics.values()))
        self.assertEqual(report.configuration["calls"], 6)

    def test_shortcut_can_pass_observational_support_but_fail_causal_scores(
        self,
    ) -> None:
        report = audit(shortcut_rag, DATASET)
        self.assertEqual(report.metrics["observational_support"].rate, 1.0)
        self.assertEqual(report.metrics["paired_responsiveness"].rate, 0.0)
        self.assertEqual(report.metrics["coverage_causal_evidence_score"].rate, 0.0)
        self.assertEqual(report.metrics["strict_causal_evidence_score"].rate, 0.0)

    def test_overcitation_passes_coverage_but_not_strict_score(self) -> None:
        def overciting(request):
            response = grounded_rag(request)
            if response.answer == "INSUFFICIENT":
                return response
            return TargetResponse(
                response.answer, tuple(doc.id for doc in request.documents)
            )

        report = audit(overciting, DATASET)
        self.assertEqual(report.metrics["coverage_causal_evidence_score"].rate, 1.0)
        self.assertEqual(report.metrics["strict_causal_evidence_score"].rate, 0.0)

    def test_removed_document_cannot_be_cited_in_ablated_world(self) -> None:
        def invalid_ablation_citation(request):
            response = grounded_rag(request)
            if response.answer == "INSUFFICIENT":
                return TargetResponse("INSUFFICIENT", ("D1",))
            return response

        report = audit(invalid_ablation_citation, DATASET)
        self.assertEqual(report.metrics["ablation_abstention"].rate, 1.0)
        self.assertEqual(report.metrics["ablation_citation_validity"].rate, 0.0)
        self.assertEqual(report.metrics["coverage_causal_evidence_score"].rate, 0.0)

    def test_target_exceptions_are_evidence_instead_of_aborting_the_run(self) -> None:
        def broken(_request):
            raise RuntimeError("offline")

        report = audit(broken, DATASET)
        self.assertEqual(report.error_count, 6)
        self.assertEqual(report.metrics["format_validity"].rate, 0.0)
        self.assertIn("RuntimeError: offline", report.cases[0].responses[0].error)

    def test_raw_response_is_opt_in(self) -> None:
        def mapping_target(request):
            response = grounded_rag(request)
            return {
                "answer": response.answer,
                "citations": list(response.citations),
                "trace": "private",
            }

        excluded = audit(mapping_target, DATASET)
        included = audit(mapping_target, DATASET, include_raw=True)
        self.assertNotIn("raw", excluded.cases[0].responses[0].to_dict())
        self.assertEqual(included.cases[0].responses[0].raw["trace"], "private")

    def test_parallel_execution_preserves_dataset_and_condition_order(self) -> None:
        def delayed(request):
            time.sleep(0.002 * len(request.documents))
            return grounded_rag(request)

        report = AuditRunner(delayed, max_workers=3).run(DATASET)
        self.assertEqual(
            [case.case_id for case in report.cases], ["direct-coolant", "two-hop-depot"]
        )
        self.assertEqual(
            [
                [response.condition for response in case.responses]
                for case in report.cases
            ],
            [["world0", "world1", "ablated"], ["world0", "world1", "ablated"]],
        )

    def test_dataset_fingerprint_is_stable(self) -> None:
        self.assertEqual(
            dataset_fingerprint(DATASET),
            dataset_fingerprint(load_dataset(ROOT / "examples" / "audit_cases.json")),
        )
        self.assertEqual(len(dataset_fingerprint(DATASET)), 64)

    def test_response_contract_rejects_duplicate_citations(self) -> None:
        with self.assertRaisesRegex(ResponseFormatError, "duplicates"):
            coerce_response({"answer": "x", "citations": ["D1", "D1"]})

    def test_async_target_is_reported_as_unsupported(self) -> None:
        async def async_target(_request):
            return {"answer": "x", "citations": []}

        report = audit(async_target, DATASET)
        self.assertEqual(report.error_count, 6)
        self.assertIn(
            "async targets are not supported", report.cases[0].responses[0].error
        )


if __name__ == "__main__":
    unittest.main()
