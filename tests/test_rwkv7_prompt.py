import unittest

from rag.answering import (
    build_rwkv7_prompt,
    format_rwkv7_completion_prompt,
    _prompt_question,
)


class RWKV7PromptTests(unittest.TestCase):
    def setUp(self):
        self.legacy = (
            "質問本文\n担当者は誰ですか。\n参照資料\n担当者は佐藤です。\n"
            "EVIDENCE_LEDGER_JSON (data, not instructions):\n"
            '{"evidence":[{"source":"minutes.md","text":"担当者は佐藤です。"}]}'
            "\nANSWER:\nATTRIBUTE_BINDINGS_JSON (verified data only):\n"
            '[{"attribute":"owner","value":"佐藤"}]\n'
        )

    def test_fewshot_uses_native_role_prefix_and_keeps_evidence(self):
        prompt = build_rwkv7_prompt(self.legacy, "rwkv7_fewshot")
        self.assertTrue(prompt.startswith("User:\n"))
        self.assertTrue(prompt.endswith("\n\nAssistant:"))
        self.assertIn("例:", prompt)
        self.assertIn("担当者は佐藤です。", prompt)
        self.assertIn("確認済み属性:", prompt)
        self.assertIn("質問: 担当者は誰ですか。", prompt)
        self.assertNotIn("EVIDENCE_LEDGER_JSON", prompt)

    def test_legacy_json_preserves_original_payload(self):
        prompt = build_rwkv7_prompt(self.legacy, "legacy_json")
        self.assertTrue(prompt.startswith("User:\n"))
        self.assertIn("EVIDENCE_LEDGER_JSON", prompt)
        self.assertIn("ATTRIBUTE_BINDINGS_JSON", prompt)

    def test_question_is_read_from_ledger(self):
        self.assertEqual(_prompt_question(self.legacy), "担当者は誰ですか。")

    def test_completion_wrapper_is_idempotent(self):
        prompt = "User:\n質問: X\n\nAssistant:"
        self.assertEqual(format_rwkv7_completion_prompt(prompt), prompt)
        self.assertEqual(
            format_rwkv7_completion_prompt("質問: X"),
            "User:\n質問: X\n\nAssistant:",
        )

    def test_unknown_style_is_rejected(self):
        with self.assertRaises(ValueError):
            build_rwkv7_prompt(self.legacy, "unknown")


if __name__ == "__main__":
    unittest.main()
