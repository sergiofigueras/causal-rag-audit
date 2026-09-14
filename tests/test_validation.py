from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from causal_rag_audit import DatasetValidationError, dataset_from_mapping, load_dataset

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "audit_cases.json"


def example_mapping() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


class DatasetValidationTests(unittest.TestCase):
    def assert_invalid(self, mapping: dict, message: str) -> None:
        with self.assertRaisesRegex(DatasetValidationError, message):
            dataset_from_mapping(mapping)

    def test_example_is_valid_and_ablation_removes_only_declared_documents(
        self,
    ) -> None:
        dataset = load_dataset(EXAMPLE)
        self.assertEqual(dataset.name, "two-case-demo")
        self.assertEqual(len(dataset.cases), 2)
        self.assertEqual(
            [doc.id for doc in dataset.cases[0].documents_for("ablated")], ["D2"]
        )

    def test_worlds_require_distinct_oracle_answers(self) -> None:
        mapping = example_mapping()
        mapping["cases"][0]["answers"]["world1"] = " argon x "
        self.assert_invalid(mapping, "oracle answers must differ")

    def test_changed_documents_must_be_declared_exactly(self) -> None:
        mapping = example_mapping()
        mapping["cases"][0]["world1"][1]["text"] = "A silent undeclared change."
        self.assert_invalid(mapping, "do not match declarations")

    def test_nuisance_document_cannot_be_in_an_oracle_proof(self) -> None:
        mapping = example_mapping()
        mapping["cases"][0]["world1"][1]["text"] = "A declared nuisance change."
        mapping["cases"][0]["nuisance_change_ids"] = ["D2"]
        mapping["cases"][0]["proofs"]["world0"].append("D2")
        self.assert_invalid(mapping, "nuisance_change_ids cannot participate")

    def test_ablation_must_remove_world0_proof_evidence(self) -> None:
        mapping = example_mapping()
        mapping["cases"][0]["ablate_document_ids"] = ["D2"]
        self.assert_invalid(mapping, "must remove world0 proof documents")

    def test_document_ids_must_be_unique_within_a_world(self) -> None:
        mapping = example_mapping()
        mapping["cases"][0]["world0"][1]["id"] = "D1"
        self.assert_invalid(mapping, "document IDs must be unique")

    def test_answer_aliases_cannot_overlap_between_worlds(self) -> None:
        mapping = example_mapping()
        mapping["cases"][0]["answer_aliases"]["world1"] = ["ARGON X"]
        self.assert_invalid(mapping, "answer aliases overlap")

    def test_unknown_fields_are_rejected(self) -> None:
        mapping = copy.deepcopy(example_mapping())
        mapping["cases"][0]["oracle_leak"] = True
        self.assert_invalid(mapping, "unknown fields")


if __name__ == "__main__":
    unittest.main()
