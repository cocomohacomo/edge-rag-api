import tempfile, unittest
from pathlib import Path
from rag.core import Store, chunks, japanese_terms, answer_from_context
from rag.answering import (
    ABSTAIN, answer_with_pipeline, assemble_evidence_answer, build_evidence_prompt, build_answer_plan,
    build_progress_summary, extractive_fallback, local_llama_generator,
    retrieval_gate, validate_answer, generation_budget,
)


class CoreTests(unittest.TestCase):
    def test_japanese_bigrams(self):
        self.assertIn('人工', japanese_terms('人工知能'))
        self.assertIn('知能', japanese_terms('人工知能'))

    def test_chunk_overlap(self):
        parts = chunks('あ' * 1000, 300, 50)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(parts))

    def test_ingest_search_and_history(self):
        with tempfile.TemporaryDirectory() as td:
            db = Store(Path(td) / 'test.db')
            p = Path(td) / 'manual.txt'
            p.write_text('SURGEは情報をクラスターとして管理する。フォーカスは認知対象を示す。', encoding='utf-8')
            self.assertGreater(db.add_document(p, p.read_text(encoding='utf-8')), 0)
            hits = db.search('情報 クラスター', 3)
            self.assertTrue(hits)
            self.assertIn('manual.txt', answer_from_context('クラスターとは', hits))
            db.log('user', 'テスト')
            self.assertEqual(db.history(1)[0]['content'], 'テスト')
            db.close()

    def test_retrieval_gate_rejects_unrelated_documents(self):
        hits = [
            {'score': .3, 'source': 'index.md', 'text': '文書をチャンク化してFTS5に登録する。'},
            {'score': .2, 'source': 'offline.md', 'text': '初回導入後はオフラインで動作する。'},
        ]
        decision = retrieval_gate('文書をチャンク化してFTS5に登録する方法は？', hits)
        self.assertEqual([h['source'] for h in decision.accepted], ['index.md'])
        self.assertEqual([h['source'] for h in decision.rejected], ['offline.md'])

    def test_gate_abstains_when_no_evidence(self):
        result = answer_with_pipeline('量子暗号の方式は？', [
            {'score': .2, 'source': 'a', 'text': '会議担当者は佐藤。期限は12月15日。'}
        ])
        self.assertEqual(result['status'], 'abstained')
        self.assertEqual(result['answer'], ABSTAIN)

    def test_evidence_prompt_is_constrained(self):
        prompt = build_evidence_prompt('担当者は？', [
            {'source': 'minutes.md', 'text': '担当者は佐藤。'}
        ])
        self.assertIn('根拠だけ', prompt)
        self.assertIn('minutes.md', prompt)
        self.assertIn('担当者は？', prompt)

    def test_validator_rejects_unsupported_number_and_empty(self):
        hits = [{'source': 'minutes.md', 'text': '担当者は佐藤。期限は12月15日。'}]
        self.assertEqual(validate_answer('', '期限は？', hits), (False, 'empty'))
        self.assertEqual(validate_answer('期限は12月20日です。', '期限は？', hits)[1],
                         'unsupported_number')

    def test_validator_accepts_grounded_answer(self):
        hits = [{'source': 'minutes.md', 'text': '担当者は佐藤。期限は12月15日。'}]
        ok, _ = validate_answer('担当者は佐藤で、期限は12月15日です。', '担当者と期限は？', hits)
        self.assertTrue(ok)

    def test_pipeline_retries_then_uses_extractive_fallback(self):
        hits = [{'score': .4, 'source': 'minutes.md',
                 'text': '会議の担当者は佐藤です。提出期限は12月15日です。'}]
        calls = []
        def broken(_prompt):
            calls.append(1)
            return '' if len(calls) == 1 else '全く関係のない回答です。'
        result = answer_with_pipeline('会議の担当者と提出期限は？', hits, broken, retry=True, continuation=False, prefer_evidence_assembly=False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['status'], 'evidence_assembled')
        self.assertIn('佐藤', result['answer'])
        self.assertIn('minutes.md', result['answer'])

    def test_llm_adapter_rejects_non_loopback_endpoint(self):
        with self.assertRaises(ValueError):
            local_llama_generator('http://example.com:8080')

    def test_llm_adapter_accepts_loopback_endpoint(self):
        self.assertTrue(callable(local_llama_generator('http://127.0.0.1:8080')))

    def test_extractive_fallback_returns_abstention_without_match(self):
        self.assertEqual(extractive_fallback('量子暗号の方式は？', []), ABSTAIN)

    def test_pipeline_defaults_to_single_generation_attempt(self):
        hits = [{'score': .4, 'source': 'minutes.md',
                 'text': '会議の担当者は佐藤です。提出期限は12月15日です。'}]
        calls = []
        result = answer_with_pipeline('会議の担当者と提出期限は？', hits,
                                      lambda _prompt: calls.append(1) or '', continuation=False)
        self.assertEqual(len(calls), 0)
        self.assertEqual(result['status'], 'evidence_assembled')

    def test_validator_rejects_internal_model_tags(self):
        hits = [{'source': 'offline.md', 'text': '初回導入後はオフラインで動作する。'}]
        self.assertEqual(validate_answer('</system_thinking> オフラインで動作します。',
                                         'ネットワーク要件は？', hits)[1], 'internal_tag')


    def test_generation_budget_scales_with_question_shape(self):
        self.assertEqual(generation_budget('期限はいつですか？'), 32)
        self.assertEqual(generation_budget('担当者と期限を教えてください。'), 48)

    def test_validator_rejects_token_limit_completion(self):
        hits = [{'source': 'spec.md', 'text': '端末はi5-10310Uで、メモリは8GB RAM。'}]
        ok, reason = validate_answer('端末はi5-10310Uで、メモリは', '端末とメモリは？',
                                     hits, {'stopped_limit': True, 'tokens_predicted': 48, 'n_predict': 48})
        self.assertFalse(ok)
        self.assertEqual(reason, 'token_limit')

    def test_pipeline_falls_back_on_truncated_generation(self):
        hits = [{'score': .4, 'source': 'spec.md',
                 'text': '対象端末はIntel i5-10310U。メモリ要件は8GB RAM。'}]
        result = answer_with_pipeline('端末とメモリ要件は？', hits,
            lambda _prompt: {'content': '対象端末はIntel i5-10310Uで、メモリは',
                             'stop': False, 'tokens_predicted': 48, 'n_predict': 48}, prefer_evidence_assembly=False)
        self.assertEqual(result['status'], 'evidence_assembled')
        self.assertIn('8GB', result['answer'])


    def test_semantic_continuation_completes_missing_deadline(self):
        hits = [{'score': .4, 'source': 'minutes.md',
                 'text': '会議の担当者は佐藤です。提出期限は12月15日です。'}]
        calls = []
        def generator(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                return {'content': '担当者は佐藤です。<CONTINUE>',
                        'stop': False, 'tokens_predicted': 48, 'n_predict': 48}
            return {'content': '提出期限は12月15日です。<END_ANSWER>',
                    'stop': True, 'tokens_predicted': 12, 'n_predict': 24}
        result = answer_with_pipeline('会議の担当者と提出期限は？', hits, generator, prefer_evidence_assembly=False)
        self.assertEqual(result['status'], 'generated')
        self.assertIn('佐藤', result['answer'])
        self.assertIn('12月15日', result['answer'])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result['generation_rounds']), 2)
        self.assertIn('担当者は佐藤です。', calls[1])

    def test_semantic_continuation_stops_at_round_limit(self):
        hits = [{'score': .4, 'source': 'minutes.md',
                 'text': '会議の担当者は佐藤です。提出期限は12月15日です。'}]
        calls = []
        def generator(_prompt):
            calls.append(1)
            return {'content': '担当者は佐藤です。<CONTINUE>',
                    'stop': True, 'tokens_predicted': 10, 'n_predict': 24}
        result = answer_with_pipeline('会議の担当者と提出期限は？', hits, generator,
                                      max_continuations=1, prefer_evidence_assembly=False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['status'], 'evidence_assembled')

    def test_continuation_marker_is_not_exposed(self):
        hits = [{'score': .4, 'source': 'offline.md',
                 'text': '初回導入後はオフラインで動作する。'}]
        result = answer_with_pipeline('初回導入後の動作は？', hits,
            lambda _prompt: '初回導入後はオフラインで動作します。<END_ANSWER>')
        self.assertNotIn('<END_ANSWER>', result['answer'])
        self.assertEqual(result['status'], 'generated')


    def test_complete_answer_does_not_continue_on_token_limit_marker(self):
        hits = [{'score': .4, 'source': 'offline.md',
                 'text': '初回導入後はオフラインで動作する。'}]
        calls = []
        def generator(_prompt):
            calls.append(1)
            return {'content': '初回導入後はオフラインで動作します。<CONTINUE>',
                    'stopped_limit': True, 'tokens_predicted': 48, 'n_predict': 48}
        result = answer_with_pipeline('初回導入後の動作は？', hits, generator)
        self.assertEqual(result['status'], 'generated')
        self.assertEqual(len(calls), 1)
        self.assertNotIn('<CONTINUE>', result['answer'])

    def test_unsupported_partial_answer_does_not_trigger_continuation(self):
        hits = [{'score': .4, 'source': 'audio.md',
                 'text': 'CPUのみでローカル実行する方式を採用する。'}]
        calls = []
        def generator(_prompt):
            calls.append(1)
            return {'content': '今日は晴れです。<CONTINUE>',
                    'stopped_limit': True, 'tokens_predicted': 48, 'n_predict': 48}
        result = answer_with_pipeline('CPUのみで動く実行方式は？', hits, generator)
        self.assertEqual(result['status'], 'evidence_assembled')
        self.assertEqual(len(calls), 0)
        self.assertIn('CPU', result['answer'])


    def test_complete_sentence_with_missing_slot_uses_fallback_without_continuation(self):
        hits = [{'score': .4, 'source': 'minutes.md',
                 'text': '会議の担当者は佐藤です。提出期限は後日調整する。'}]
        calls = []
        def generator(_prompt):
            calls.append(1)
            return {'content': '担当者は佐藤です。<CONTINUE>',
                    'stopped_limit': True, 'tokens_predicted': 32, 'n_predict': 32}
        result = answer_with_pipeline('会議の担当者と提出期限は？', hits, generator)
        self.assertEqual(result['status'], 'extractive_fallback')
        self.assertEqual(len(calls), 1)

    def test_truncated_grounded_sentence_continues_without_control_marker(self):
        hits = [{'score': .4, 'source': 'offline.md',
                 'text': '初回導入後はオフラインで動作する。'}]
        calls = []
        def generator(_prompt):
            calls.append(1)
            if len(calls) == 1:
                return {'content': '初回導入後はオフラインで',
                        'stop': False, 'tokens_predicted': 24, 'n_predict': 24}
            return {'content': '動作します。<END_ANSWER>',
                    'stop': True, 'tokens_predicted': 8, 'n_predict': 24}
        result = answer_with_pipeline('初回導入後の動作は？', hits, generator)
        self.assertEqual(result['status'], 'generated')
        self.assertEqual(len(calls), 2)
        self.assertIn('オフラインで動作します', result['answer'])
        self.assertTrue(result['generation_rounds'][0]['truncated'])

    def test_evidence_prompt_uses_real_newlines(self):
        prompt = build_evidence_prompt('担当者は？', [
            {'source': 'minutes.md', 'text': '担当者は佐藤です。'}
        ])
        self.assertIn('\n', prompt)
        self.assertNotIn('\\n', prompt)


    def test_answer_plan_maps_required_attributes_to_source_evidence(self):
        hits = [{
            "score": .4, "source": "spec.md",
            "text": "担当者は佐藤です。期限は12月15日です。URLはhttps://example.com/docs。パスはdocs/spec.mdです。"
        }]
        plan = build_answer_plan("担当者と期限、URL、パスは？", hits)
        self.assertEqual(plan["requirements"], ["owner", "deadline", "url", "path"])
        self.assertEqual(plan["missing_requirements"], [])
        self.assertIn("佐藤", plan["evidence_mapping"]["owner"][0]["values"])
        self.assertIn("12月15日", plan["evidence_mapping"]["deadline"][0]["values"])
        self.assertEqual(plan["evidence_mapping"]["url"][0]["values"][0], "https://example.com/docs")
        self.assertIn("docs/spec.md", plan["evidence_mapping"]["path"][0]["values"])

    def test_evidence_assembly_builds_answer_from_all_supported_requirements(self):
        hits = [{
            "score": .4, "source": "minutes.md",
            "text": "担当者は佐藤です。期限は12月15日です。URLはhttps://example.com/docs。パスはdocs/spec.mdです。"
        }]
        plan = build_answer_plan("担当者と期限、URL、パスは？", hits)
        answer = assemble_evidence_answer(plan)
        self.assertIn("担当者: 佐藤", answer)
        self.assertIn("期限: 12月15日", answer)
        self.assertIn("https://example.com/docs", answer)
        self.assertIn("docs/spec.md", answer)
        self.assertIn("minutes.md", answer)

    def test_evidence_assembly_refuses_incomplete_plans(self):
        plan = build_answer_plan("担当者と期限は？", [
            {"score": .4, "source": "minutes.md", "text": "担当者は佐藤です。"}
        ])
        self.assertIsNone(assemble_evidence_answer(plan))

    def test_pipeline_uses_evidence_assembly_when_generator_unavailable(self):
        hits = [{
            "score": .4, "source": "minutes.md",
            "text": "担当者は佐藤です。期限は12月15日です。"
        }]
        result = answer_with_pipeline("会議の担当者と期限を教えてください", hits, None)
        self.assertEqual(result["status"], "evidence_assembled")
        self.assertIn("佐藤", result["answer"])
        self.assertIn("12月15日", result["answer"])
        self.assertEqual(result["generation_rounds"], 0)

    def test_answer_plan_tracks_missing_evidence_without_inventing_values(self):
        plan = build_answer_plan("担当者と期限は？", [
            {"score": .4, "source": "minutes.md", "text": "担当者は佐藤です。"}
        ])
        self.assertEqual(plan["missing_requirements"], ["deadline"])
        self.assertFalse(plan["validation"]["requirements_complete"])

    def test_validator_rejects_unsupported_url_and_path(self):
        hits = [{"source": "spec.md", "text": "URLはhttps://example.com/docs。パスはdocs/spec.mdです。"}]
        self.assertEqual(validate_answer("URLはhttps://evil.example/docsです。", "URLは？", hits)[1],
                         "unsupported_url")
        self.assertEqual(validate_answer("パスはprivate/secret.txtです。", "パスは？", hits)[1],
                         "unsupported_path")

    def test_progress_summary_only_marks_validation_after_it_runs(self):
        plan = build_answer_plan("担当者は？", [
            {"score": .4, "source": "minutes.md", "text": "担当者は佐藤です。"}
        ])
        pending = build_progress_summary(plan)
        complete = build_progress_summary(plan, validated=True)
        self.assertFalse(pending["checks"][-1]["done"])
        self.assertTrue(complete["checks"][-1]["done"])

    def test_evidence_prompt_explicitly_plans_requirements_and_preserves_identifiers(self):
        hits = [{
            "source": "spec.md",
            "text": "担当者は佐藤です。期限は12月15日です。URLはhttps://example.com/docs。パスはdocs/spec.mdです。"
        }]
        prompt = build_evidence_prompt("担当者と期限、URL、パスは？", hits)
        self.assertIn("要求属性:", prompt)
        self.assertIn("owner", prompt)
        self.assertIn("deadline", prompt)
        self.assertIn("URL", prompt)
        self.assertIn("パス", prompt)
        self.assertIn("https://example.com/docs", prompt)
        self.assertIn("docs/spec.md", prompt)
        self.assertIn("文字列を改変せず正確に転記", prompt)


    def test_generation_exception_is_recorded_without_retry_loop(self):
        hits = [{'score': .4, 'source': 'offline.md',
                 'text': '初回導入後はオフラインで動作する。'}]
        calls = []
        def broken(_prompt):
            calls.append(1)
            raise TimeoutError("local model timed out")
        result = answer_with_pipeline('初回導入後の動作は？', hits, broken,
                                      retry=True, continuation=True, max_continuations=2,
                                      prefer_evidence_assembly=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['status'], 'extractive_fallback')
        self.assertEqual(result['generation_rounds'][0]['state'], 'generation_error')
        self.assertEqual(result['reason'], 'generation_error')

    def test_generation_loop_records_grounded_continuation_states(self):
        hits = [{'score': .4, 'source': 'minutes.md',
                 'text': '会議の担当者は佐藤です。提出期限は12月15日です。'}]
        calls = []
        def generator(_prompt):
            calls.append(1)
            if len(calls) == 1:
                return {'content': '担当者は佐藤です。<CONTINUE>',
                        'stopped_limit': True, 'tokens_predicted': 48, 'n_predict': 48}
            return {'content': '提出期限は12月15日です。<END_ANSWER>',
                    'stop': True, 'tokens_predicted': 12, 'n_predict': 24}
        result = answer_with_pipeline('会議の担当者と提出期限は？', hits, generator,
                                      prefer_evidence_assembly=False)
        self.assertEqual(result['status'], 'generated')
        self.assertEqual([r['state'] for r in result['generation_rounds']],
                         ['continue', 'completed'])
        self.assertTrue(result['generation_rounds'][0]['truncated'])

    def test_device_slot_preserves_cpu_model_suffix(self):
        hits = [{'score': .4, 'source': 'spec.md',
                 'text': '対象端末のCPUはIntel i5-10310U。'}]
        plan = build_answer_plan('対象端末のCPUは？', hits)
        self.assertIn('i5-10310U', plan['evidence_mapping']['device'][0]['values'])

    def test_relation_plan_requires_model_and_search_db_coexistence(self):
        hits = [{"score": .4, "source": "memory.md",
                 "text": "8GB RAM内でモデルと検索DBを共存させる。回答生成時のピークメモリを測定する。"}]
        plan = build_answer_plan("回答生成時に重視するメモリ制約は何ですか？", hits)
        self.assertEqual(plan["missing_relations"], [])
        answer = assemble_evidence_answer(plan)
        self.assertIn("モデルと検索DBを共存", answer)
        from rag.answering import _missing_relations
        self.assertEqual(_missing_relations("回答生成時に重視するメモリ制約は何ですか？",
                                            "メモリは8GB RAMです。", hits),
                         ["relation:model_search_db_coexistence"])
        self.assertEqual(_missing_relations("回答生成時に重視するメモリ制約は何ですか？",
                                            answer, hits), [])

    def test_relation_plan_rejects_missing_relation_evidence(self):
        plan = build_answer_plan("回答生成時に重視するメモリ制約は何ですか？", [
            {"score": .4, "source": "memory.md", "text": "メモリは8GB RAM。"}
        ])
        self.assertEqual(plan["missing_relations"], ["model_search_db_coexistence"])
        self.assertIsNone(assemble_evidence_answer(plan))

    def test_offline_relation_is_tracked_separately_from_network_attribute(self):
        hits = [{"score": .4, "source": "offline.md",
                 "text": "初回導入後はオフラインで動作する。実行時のネットワーク接続を必要としない。"}]
        plan = build_answer_plan("配布時のネットワーク要件はどうなっていますか？", hits)
        self.assertEqual(plan["missing_relations"], [])
        self.assertIsNotNone(assemble_evidence_answer(plan))


    def test_forced_generation_bypasses_evidence_assembly(self):
        hits = [{"score": .9, "source": "search.md", "text": "検索方式はSQLite FTS5を使用する。"}]
        calls = []
        def generator(prompt):
            calls.append(prompt)
            return {"content": "検索方式はSQLite FTS5です。<END_ANSWER>",
                    "tokens_predicted": 8, "n_predict": 32, "stop": True}
        result = answer_with_pipeline(
            "検索方式は何ですか？", hits, generator,
            continuation=False, force_generation=True,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["generation_rounds"][0]["state"], "completed")

    def test_forced_generation_fails_closed_without_generator(self):
        hits = [{"score": .9, "source": "search.md", "text": "検索方式はSQLite FTS5を使用する。"}]
        result = answer_with_pipeline(
            "検索方式は何ですか？", hits, None, force_generation=True,
        )
        self.assertEqual(result["status"], "generation_unavailable")
        self.assertEqual(result["reason"], "forced_generation_requires_generator")

    def test_evidence_assembly_remains_separate_from_llm_generation(self):
        hits = [{"score": .9, "source": "search.md", "text": "検索方式はSQLite FTS5を使用する。"}]
        calls = []
        def generator(prompt):
            calls.append(prompt)
            return "検索方式はSQLite FTS5です。"
        result = answer_with_pipeline(
            "検索方式は何ですか？", hits, generator,
            prefer_evidence_assembly=True,
        )
        self.assertEqual(result["status"], "evidence_assembled")
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
