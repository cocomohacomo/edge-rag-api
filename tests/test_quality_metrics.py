import unittest
from rag.quality_metrics import evaluate_attribute_completeness, validate_attribute_claims
from rag.attribute_validation import build_composer_prompt, validate_composed_answer
from rag.answering import ABSTAIN, answer_with_pipeline


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

    def test_failed_final_gate_quarantines_placeholder_candidate(self):
        hits = [{
            "source": "meeting.txt",
            "text": "担当者は佐藤です。期限は12月15日です。",
            "score": 1.0,
        }]
        result = answer_with_pipeline(
            "担当者と期限は？",
            hits,
            generator=lambda prompt: "担当者は{owner}です。期限は12月15日です。",
            force_generation=True,
            continuation=False,
        )
        self.assertNotEqual(result["status"], "generated")
        self.assertEqual(result["answer"], ABSTAIN)
        self.assertTrue(result["candidate_diagnostics"]["quarantined"])
        self.assertIn("{owner}", result["candidate_diagnostics"]["candidate"])
        self.assertFalse(result["final_output_gate"]["passed"])
        self.assertFalse(result["final_output_gate"]["checks"]["attributes_valid"])

    def test_deterministic_evidence_assembly_uses_final_output_gate(self):
        hits = [{
            "source": "meeting.txt",
            "text": "担当者は佐藤です。期限は12月15日です。",
            "score": 1.0,
        }]
        result = answer_with_pipeline(
            "担当者と期限は？",
            hits,
            generator=None,
            force_generation=False,
            prefer_evidence_assembly=True,
        )
        self.assertEqual(result["status"], "evidence_assembled")
        self.assertTrue(result["final_output_gate"]["passed"])
        self.assertIn("佐藤", result["answer"])
        self.assertIn("12月15日", result["answer"])

    def test_retry_candidate_must_pass_the_same_final_output_gate(self):
        hits = [{
            "source": "meeting.txt",
            "text": "担当者は佐藤です。期限は12月15日です。",
            "score": 1.0,
        }]
        responses = iter(["", "担当者は佐藤です。期限は12月15日です。"])
        result = answer_with_pipeline(
            "担当者と期限は？",
            hits,
            generator=lambda prompt: next(responses),
            retry=True,
            continuation=False,
            force_generation=True,
        )
        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["answer"], "担当者は佐藤です。期限は12月15日です。")
        self.assertTrue(result["final_output_gate"]["passed"])
        self.assertTrue(result["generation_rounds"][-1]["retry"])

    def test_retry_token_limit_cannot_be_accepted_as_complete(self):
        hits = [{
            "source": "meeting.txt",
            "text": "担当者は佐藤です。期限は12月15日です。",
            "score": 1.0,
        }]
        responses = iter([
            "",
            {"content": "担当者は佐藤です。期限は12月15日です。",
             "stop_type": "limit", "truncated": True},
        ])
        result = answer_with_pipeline(
            "担当者と期限は？",
            hits,
            generator=lambda prompt: next(responses),
            retry=True,
            continuation=False,
            force_generation=True,
        )
        self.assertEqual(result["answer"], ABSTAIN)
        self.assertNotEqual(result["status"], "generated")
        self.assertFalse(result["final_output_gate"]["passed"])
        self.assertTrue(result["generation_rounds"][-1]["truncated"])

    def test_non_forced_generation_cannot_leak_failed_candidate(self):
        hits = [{
            "source": "meeting.txt",
            "text": "担当者は佐藤です。期限は12月15日です。",
            "score": 1.0,
        }]
        result = answer_with_pipeline(
            "担当者と期限は？",
            hits,
            generator=lambda prompt: "担当者は{owner}です。期限は12月15日です。",
            force_generation=False,
            prefer_evidence_assembly=False,
            continuation=False,
        )
        # The failed model candidate must never escape. In non-forced mode,
        # however, a separately assembled evidence answer is an allowed recovery
        # path when it independently passes the exact same final output gate.
        self.assertNotIn("{owner}", result["answer"])
        self.assertNotIn("担当者は{owner}", result["answer"])
        if result["status"] in {"evidence_assembled", "extractive_fallback"}:
            self.assertTrue(result["final_output_gate"]["passed"])
            self.assertIn("佐藤", result["answer"])
            self.assertIn("12月15日", result["answer"])
        else:
            self.assertEqual(result["status"], "abstained")
            self.assertEqual(result["answer"], ABSTAIN)
            self.assertFalse(result["final_output_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
