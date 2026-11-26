import unittest
from rag.quality_metrics import evaluate_attribute_completeness, validate_attribute_claims


class AttributeCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.ledger={"attributes":[{"attribute":"owner","status":"supported","values":["佐藤"]},{"attribute":"deadline","status":"supported","values":["12月15日"]}]}
        self.validation={"supported":["owner","deadline"],"insufficient":[],"has_conflict":False}
    def test_complete_attributes_are_scored(self):
        result=evaluate_attribute_completeness(self.ledger,self.validation,"担当者: 佐藤。期限: 12月15日。")
        self.assertEqual(result["completeness_rate"],1.0)
        self.assertEqual(result["missing_supported_attributes"],[])
    def test_missing_attribute_is_detected(self):
        result=evaluate_attribute_completeness(self.ledger,self.validation,"担当者: 佐藤。")
        self.assertEqual(result["completeness_rate"],0.5)
        self.assertEqual(result["missing_supported_attributes"],["deadline"])
    def test_claim_validator_rejects_unsupported_attribute_values(self):
        ledger={"attributes":[{"attribute":"budget","status":"insufficient","values":["100万円"]}]}
        validation={"supported":[],"insufficient":["budget"],"has_conflict":False}
        self.assertFalse(validate_attribute_claims("予算は100万円です",ledger,validation)["valid"])
    def test_conflict_fails_closed(self):
        validation={"supported":["owner"],"insufficient":[],"has_conflict":True}
        self.assertFalse(validate_attribute_claims("佐藤",self.ledger,validation)["valid"])
if __name__ == "__main__": unittest.main()
