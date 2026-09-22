"""CEA-Extended schema v2 and full intervention scoring."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from causal_rag_audit import (
    AuditRunner,
    DatasetValidationError,
    TargetResponse,
    dataset_from_mapping,
)
from causal_rag_audit.extended import ExtendedDataset
from causal_rag_audit.reports import report_as_markdown

FIXTURE = Path(__file__).resolve().parents[1] / "datasets" / "cea-extended-demo.json"


def valid_target(request):
    docs = {document.id: document for document in request.documents}
    if "D1" not in docs:
        return TargetResponse(answer="INSUFFICIENT", citations=())
    if "Bramblecross" in docs["D1"].text:
        return TargetResponse(answer="Hesper", citations=("D1", "D4"))
    if "Cedarholm" in docs["D1"].text:
        return TargetResponse(answer="Orphic", citations=("D1", "D5"))
    if "D6" in docs and "routed to Bramblecross" in docs["D6"].text:
        return TargetResponse(answer="Galdite", citations=("D1", "D3", "D6"))
    return TargetResponse(answer="Galdite", citations=("D1", "D3"))


class ExtendedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def condition(self, family):
        return next(
            item
            for item in self.raw["cases"][0]["conditions"]
            if item["family"] == family
        )

    def test_fixture_round_trip_and_complete_pass(self):
        dataset = dataset_from_mapping(self.raw)
        self.assertIsInstance(dataset, ExtendedDataset)
        self.assertEqual(
            dataset_from_mapping(dataset.to_dict()).to_dict(), dataset.to_dict()
        )
        report = AuditRunner(valid_target, scoring_profile="strict-exact").run(dataset)
        self.assertEqual(report.configuration["calls"], 7)
        self.assertEqual(report.report_schema_version, 2)
        self.assertTrue(
            report.metrics["extended_coverage_causal_evidence_score"].rate == 1
        )
        self.assertTrue(
            report.metrics["extended_strict_causal_evidence_score"].rate == 1
        )
        self.assertEqual(report.condition_coverage["counterfactual"], 2)
        self.assertEqual(report.condition_coverage["redundancy"], 1)
        diagnostics = report.cases[0].metadata["condition_diagnostics"]
        self.assertEqual(
            diagnostics["baseline"]["proof_sets"], [["D1", "D2"], ["D1", "D3"]]
        )
        self.assertEqual(diagnostics["baseline"]["matched_proof_set"], ["D1", "D3"])
        self.assertIn("## Condition diagnostics", report_as_markdown(report))

    def test_adversarial_memorized_answer_fails(self):
        dataset = dataset_from_mapping(self.raw)

        def memorized(_request):
            return {"answer": "Galdite", "citations": ["D1", "D2"]}

        report = AuditRunner(memorized, scoring_profile="strict-exact").run(dataset)
        self.assertFalse(
            report.cases[0].indicators["extended_coverage_causal_evidence_score"]
        )
        self.assertFalse(report.cases[0].indicators["counterfactual_coverage_pass"])
        self.assertFalse(report.cases[0].indicators["ablation_coverage_pass"])

    def test_abstention_citation_separates_coverage_from_strict(self):
        dataset = dataset_from_mapping(self.raw)

        def cited_abstention(request):
            if "D1" not in {doc.id for doc in request.documents}:
                return TargetResponse(answer="INSUFFICIENT", citations=("D2",))
            return valid_target(request)

        report = AuditRunner(cited_abstention, scoring_profile="strict-exact").run(
            dataset
        )
        self.assertTrue(
            report.cases[0].indicators["extended_coverage_causal_evidence_score"]
        )
        self.assertFalse(
            report.cases[0].indicators["extended_strict_causal_evidence_score"]
        )
        self.assertTrue(report.cases[0].indicators["ablation_coverage_pass"])
        self.assertFalse(report.cases[0].indicators["ablation_strict_pass"])

    def test_format_and_invalid_citation_fail_joint_with_diagnostics(self):
        dataset = dataset_from_mapping(self.raw)

        def malformed(request):
            value = valid_target(request)
            if "D1" in {doc.id for doc in request.documents} and "D2" in {
                doc.id for doc in request.documents
            }:
                return TargetResponse(
                    answer=value.answer,
                    citations=value.citations,
                    format_valid=False,
                    format_error="bad envelope",
                )
            return value

        report = AuditRunner(malformed, scoring_profile="strict-exact").run(dataset)
        self.assertFalse(report.cases[0].indicators["extended_format_validity"])
        self.assertFalse(
            report.cases[0].indicators["extended_strict_causal_evidence_score"]
        )
        self.assertGreater(report.error_count, 0)

        def bad_citation(request):
            value = valid_target(request)
            if "D1" in {doc.id for doc in request.documents}:
                return TargetResponse(
                    answer=value.answer, citations=(*value.citations, "unknown")
                )
            return value

        report = AuditRunner(bad_citation, scoring_profile="strict-exact").run(dataset)
        self.assertFalse(report.cases[0].responses[0].citation_valid)
        self.assertFalse(
            report.cases[0].indicators["extended_coverage_causal_evidence_score"]
        )

    def test_custom_judge_alternatives_exclude_same_answer(self):
        seen = []

        def judge(actual, expected, aliases, alternatives):
            seen.append((expected, alternatives))
            return actual == expected

        AuditRunner(
            valid_target, scoring_profile="strict-exact", answer_judge=judge
        ).run(dataset_from_mapping(self.raw))
        baseline = next(
            alternatives for expected, alternatives in seen if expected == "Galdite"
        )
        self.assertNotIn("Galdite", baseline)
        self.assertIn("Hesper", baseline)

    def test_structural_rejections(self):
        mutations = []

        def bad_ablation(raw):
            condition = next(
                c for c in raw["cases"][0]["conditions"] if c["family"] == "ablation"
            )
            baseline = raw["cases"][0]["conditions"][0]
            condition["documents"] = [
                doc for doc in baseline["documents"] if doc["id"] != "D2"
            ]
            condition["removed_document_ids"] = ["D2"]

        mutations.append(bad_ablation)

        def bad_redundancy(raw):
            condition = next(
                c for c in raw["cases"][0]["conditions"] if c["family"] == "redundancy"
            )
            condition["documents"] = [
                doc for doc in condition["documents"] if doc["id"] != "D1"
            ]
            condition["removed_document_ids"] = ["D1", "D2"]
            condition["proof_sets"] = []

        mutations.append(bad_redundancy)

        def bad_antichain(raw):
            raw["cases"][0]["conditions"][0]["proof_sets"].append(["D1", "D2", "D3"])

        mutations.append(bad_antichain)

        def undeclared_edit(raw):
            condition = next(
                c for c in raw["cases"][0]["conditions"] if c["family"] == "nuisance"
            )
            condition["changed_document_ids"] = []

        mutations.append(undeclared_edit)

        def missing_family(raw):
            raw["cases"][0]["conditions"] = [
                c for c in raw["cases"][0]["conditions"] if c["family"] != "conflict"
            ]

        mutations.append(missing_family)

        def bad_nuisance_answer(raw):
            self_condition = next(
                c for c in raw["cases"][0]["conditions"] if c["family"] == "nuisance"
            )
            self_condition["expected_answer"] = "Hesper"

        mutations.append(bad_nuisance_answer)

        def added_id(raw):
            condition = next(
                c for c in raw["cases"][0]["conditions"] if c["family"] == "conflict"
            )
            condition["documents"].append({"id": "D7", "text": "An added source."})

        mutations.append(added_id)
        for mutate in mutations:
            with self.subTest(mutate=mutate.__name__):
                candidate = copy.deepcopy(self.raw)
                mutate(candidate)
                with self.assertRaises(DatasetValidationError):
                    dataset_from_mapping(candidate)

    def test_counterfactuals_may_share_answer_but_not_world(self):
        second = self.raw["cases"][0]["conditions"][2]
        second["expected_answer"] = "Hesper"
        self.assertIsInstance(dataset_from_mapping(self.raw), ExtendedDataset)
        second["documents"] = copy.deepcopy(
            self.raw["cases"][0]["conditions"][1]["documents"]
        )
        with self.assertRaisesRegex(
            DatasetValidationError, "duplicates another counterfactual"
        ):
            dataset_from_mapping(self.raw)

    def test_unresolved_conflict_requires_and_scores_abstention(self):
        conflict = copy.deepcopy(self.condition("conflict"))
        conflict["id"] = "unresolved-conflict"
        conflict["documents"][-1]["text"] = (
            "Current registry: Vela is routed to Bramblecross."
        )
        conflict["expected_answer"] = None
        conflict["proof_sets"] = []
        conflict["resolution_policy"] = "abstain"
        conflict["authority_document_ids"] = []
        conflict["rationale"] = (
            "Equal-priority current registries disagree; the declared policy is to abstain."
        )
        self.raw["cases"][0]["conditions"].append(conflict)
        dataset = dataset_from_mapping(self.raw)

        def conflict_target(request):
            docs = {doc.id: doc for doc in request.documents}
            if "D6" in docs and docs["D6"].text.startswith("Current registry"):
                return TargetResponse(answer="INSUFFICIENT", citations=())
            return valid_target(request)

        report = AuditRunner(conflict_target, scoring_profile="strict-exact").run(
            dataset
        )
        self.assertEqual(report.condition_coverage["conflict"], 2)
        self.assertTrue(
            report.cases[0].indicators["extended_strict_causal_evidence_score"]
        )
        conflict["resolution_policy"] = "authority"
        with self.assertRaises(DatasetValidationError):
            dataset_from_mapping(self.raw)


if __name__ == "__main__":
    unittest.main()
