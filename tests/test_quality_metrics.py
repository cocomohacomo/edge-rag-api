import unittest
from rag.quality_metrics import evaluate_attribute_completeness, validate_attribute_claims
from rag.attribute_validation import build_composer_prompt, validate_composed_answer


class AttributeCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.ledger = {"attributes": [
            {"attribute": "owner", "status": "supported", "values": ["佐藤"]},
            {"attribute": "deadline", "status": "supported", "values": ["12月15日"]},
            {"attribute": "budget", "status": "insufficient", "values": []},
        ]}
        self.validation = {"supported": ["owner", "deadline"], "insufficient": ["budget"], "has_conflict": False}

    def test_completeness_denominator_is_supported_attributes_only(self):
        result = evaluate_attribute_completeness(self.ledger, self.validation, "担当者: 佐藤。期限: 12月15日。")
        self.assertEqual(result["completeness_rate"], 1.0)
        self.assertEqual(result["supported_count"], 2)
        self.assertEqual(result["unsupported_attributes"], ["budget"])

    def test_missing_supported_attribute_is_detected(self):
        result = evaluate_attribute_completeness(self.ledger, self.validation, "担当者: 佐藤。")
        self.assertEqual(result["completeness_rate"], 0.5)
        self.assertEqual(result["missing_supported_attributes"], ["deadline"])

    def test_insufficient_value_is_rejected_when_explicitly_recorded(self):
        ledger = {"attributes": [{"attribute": "budget", "values": ["100万円"]}]}
        validation = {"supported": [], "insufficient": ["budget"], "has_conflict": False}
        self.assertFalse(validate_attribute_claims("予算は100万円です", ledger, validation)["valid"])

    def test_conflict_fails_closed(self):
        validation = {"supported": ["owner"], "insufficient": [], "has_conflict": True}
        self.assertFalse(validate_attribute_claims("佐藤", self.ledger, validation)["valid"])

    def test_empty_supported_set_has_neutral_completeness(self):
        result = evaluate_attribute_completeness(self.ledger, {"supported": [], "has_conflict": False}, "回答")
        self.assertEqual(result["completeness_rate"], 1.0)
        self.assertEqual(result["complete_count"], 0)

    def test_composer_prompt_requests_template_then_verified_substitution(self):
        prompt = build_composer_prompt("質問: 担当者と期限は？", self.ledger, self.validation)
        self.assertIn("回答文テンプレートを内部で設計", prompt)
        self.assertIn("{owner}", prompt)
        self.assertIn("{deadline}", prompt)
        self.assertIn("波括弧のプレースホルダー自体は最終回答に残さない", prompt)

    def test_unfilled_template_placeholder_is_rejected(self):
        valid, reason = validate_composed_answer(
            "担当者は{owner}です。期限は12月15日です。",
            self.ledger, self.validation,
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "unfilled_template_placeholder")


if __name__ == "__main__":
    unittest.main()
