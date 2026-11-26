import unittest
from rag.quality_metrics import evaluate_attribute_completeness, validate_attribute_claims


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


if __name__ == "__main__":
    unittest.main()
