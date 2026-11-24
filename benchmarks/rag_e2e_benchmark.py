from __future__ import annotations
import argparse, json, os, platform, random, statistics, subprocess, tempfile, time, urllib.request, urllib.error, resource
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.core import Store
from rag.answering import answer_with_pipeline, generation_budget, _missing_slots, _missing_relations
from rag.retrieval_v2 import search_with_diagnostics

CASES = [
    {"id":"agri-01","topic":"農地転用","question":"農地の現状変更で、表土を確保する際に確認すべき基準は何ですか？","answer":"表土の確保と土壌の適正","terms":["表土","土壌","現状変更"],"source":"農地現状変更運用基準.md"},
    {"id":"meeting-01","topic":"会議決定","question":"会議で決定した担当者と期限を教えてください。","answer":"佐藤が12月15日まで","terms":["佐藤","12月15日"],"source":"会議決定事項.md"},
    {"id":"system-01","topic":"端末要件","question":"このシステムの想定端末とメモリ要件は何ですか？","answer":"Intel i5-10310Uと8GB RAM","terms":["i5-10310U","8GB"],"source":"端末要件.md"},
    {"id":"rag-01","topic":"検索構成","question":"日本語文書の検索には何を使っていますか？","answer":"SQLite FTS5と日本語2-gram","terms":["SQLite FTS5","2-gram"],"source":"検索設計.md"},
    {"id":"offline-01","topic":"配布方式","question":"配布時のネットワーク要件はどうなっていますか？","answer":"初回導入後はオフラインで動作","terms":["オフライン","ネットワーク"],"source":"オフライン配布方針.md"},
    {"id":"audio-01","topic":"音声認識","question":"音声認識の実行方式とCPU要件を教えてください。","answer":"CPUのみでローカル実行","terms":["CPU","ローカル"],"source":"音声認識構成.md"},
    {"id":"memory-01","topic":"メモリ制約","question":"回答生成時に重視するメモリ制約は何ですか？","answer":"8GB RAM内でモデルと検索DBを共存","terms":["8GB","検索DB"],"source":"メモリ設計.md"},
    {"id":"index-01","topic":"索引","question":"検索インデックスはどのように構築しますか？","answer":"文書をチャンク化してFTS5に登録","terms":["チャンク","FTS5"],"source":"索引設計.md"},
]
DOCS = [
("農地現状変更運用基準.md","農地の現状変更では、施工前に現況を確認し、耕作に適した表土を確保する。搬入土の品質、土壌の適正、排水性、施工後の営農可能性を確認する。表土の確保と土壌の適正を記録し、農業委員会が現地確認する。"),
("会議決定事項.md","会議決定事項。担当者は佐藤。期限は12月15日。佐藤が12月15日までに調査結果を提出する。"),
("端末要件.md","対象端末はWindows 11搭載のIntel i5-10310U。メモリ要件は8GB RAM。外部GPUを前提としない。"),
("検索設計.md","日本語文書の全文検索にはSQLite FTS5を使用する。日本語テキストは2-gramに分割して索引化する。BM25と単語重複率で順位を再評価する。"),
("オフライン配布方針.md","初回導入後はオフラインで動作する。実行時のネットワーク接続を必要としない。モデルとデータをローカルに保持する。"),
("音声認識構成.md","音声認識はCPUのみでローカル実行する。ネットワークやGPUを必須としない。"),
("メモリ設計.md","8GB RAM内でモデルと検索DBを共存させる。回答生成時のピークメモリを測定し、検索DBとモデルのメモリ競合を避ける。"),
("索引設計.md","登録文書をチャンク化してFTS5に登録する。チャンクには元文書名と位置情報を保持する。"),
]

def get_json(url, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode())

def proc_memory_mb(pid):
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        vals = {}
        for line in status.splitlines():
            if line.startswith(("VmRSS:","VmHWM:")):
                k,v = line.split(":",1); vals[k] = int(v.strip().split()[0]) / 1024
        return vals
    except Exception:
        return {}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--server", default="llama.cpp/build/bin/llama-server")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--output", default="rag-e2e-results.json")
    args=ap.parse_args()
    server=[args.server,"-m",args.model,"--host","127.0.0.1","--port",str(args.port),"-t",str(args.threads),"-c",str(args.ctx),"-ngl","0"]
    started=time.perf_counter()
    proc=subprocess.Popen(server,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    base=f"http://127.0.0.1:{args.port}"
    try:
        ready=False
        for _ in range(180):
            if proc.poll() is not None: raise RuntimeError(f"llama-server exited: {proc.returncode}")
            try:
                get_json(base+"/health",timeout=2); ready=True; break
            except Exception: time.sleep(1)
        if not ready: raise TimeoutError("llama-server readiness timeout")
        server_startup=time.perf_counter()-started
        retrieval_ms=[]; rows=[]
        modes=("single","semantic_continuation")
        stats={mode:{"requests":[],"facts":0,"statuses":{},"prompt_ms":[],"generation_ms":[],
                     "prompt_tps":[],"generation_tps":[],"continuations":0,"rounds":[],
                     "plan_covered":0,"plan_total":0,"answer_covered":0,
                     "relation_covered":0,"relation_total":0} for mode in modes}
        retrieval_hits=0
        with tempfile.TemporaryDirectory() as td:
            store=Store(Path(td)/"rag.sqlite3")
            for name,doc_text in DOCS:
                store.add_document(Path(td)/name,doc_text)
            for case_index,case in enumerate(CASES):
                t=time.perf_counter(); hits,retrieval_diag=search_with_diagnostics(store,case["question"],12)
                retrieval_ms.append((time.perf_counter()-t)*1000)
                found=any(h["source"]==case["source"] for h in hits)
                retrieval_hits += int(found)
                # Alternate order to reduce systematic prompt-cache warm-up bias.
                order=list(modes) if case_index % 2 == 0 else list(reversed(modes))
                for mode in order:
                    call_metrics=[]
                    def generate(prompt):
                        question=prompt.split("\nQ:",1)[-1].split("\n根拠:",1)[0]
                        budget=min(24,args.max_tokens) if "[CONTINUATION]" in prompt else generation_budget(question,args.max_tokens)
                        response=get_json(base+"/completion",{
                            "prompt":prompt,"n_predict":budget,"temperature":0.0,
                            "top_k":1,"top_p":1.0,"seed":42,"cache_prompt":True
                        },timeout=300)
                        timings=response.get("timings",{})
                        pms=float(timings.get("prompt_ms",0)); gms=float(timings.get("predicted_ms",0))
                        pn=int(timings.get("prompt_n",0)); gn=int(timings.get("predicted_n",0))
                        call_metrics.append({"prompt_ms":pms,"generation_ms":gms,"prompt_tokens":pn,
                            "generated_tokens":gn,"stop":response.get("stop"),
                            "stopped_eos":response.get("stopped_eos"),
                            "stopped_word":response.get("stopped_word"),
                            "stopped_limit":response.get("stopped_limit"),"stop_type":response.get("stop_type"),"truncated":response.get("truncated"),"n_predict":budget})
                        return {"content":response.get("content") or "",
                            "stop":response.get("stop"),"stopped_eos":response.get("stopped_eos"),
                            "stopped_word":response.get("stopped_word"),
                            "stopped_limit":response.get("stopped_limit"),"stop_type":response.get("stop_type"),
                            "truncated":response.get("truncated"),"tokens_predicted":gn,"n_predict":budget}
                    t=time.perf_counter()
                    result=answer_with_pipeline(case["question"],hits,generate,
                        retry=False,continuation=(mode=="semantic_continuation"),max_continuations=2)
                    total_ms=(time.perf_counter()-t)*1000
                    answer=result["answer"]; status=result["status"]
                    mode_stats=stats[mode]
                    mode_stats["requests"].append(total_ms)
                    mode_stats["statuses"][status]=mode_stats["statuses"].get(status,0)+1
                    ok=all(term.lower() in answer.lower() for term in case["terms"])
                    mode_stats["facts"] += int(ok)
                    plan=result.get("answer_plan", {})
                    requirements=plan.get("requirements", [])
                    missing=set(plan.get("missing_requirements", []))
                    answer_missing=set(_missing_slots(case["question"], answer, result["retrieval"].accepted))
                    mode_stats["plan_total"] += len(requirements)
                    mode_stats["plan_covered"] += sum(1 for slot in requirements if slot not in missing)
                    mode_stats["answer_covered"] += sum(1 for slot in requirements if slot not in answer_missing)
                    relation_specs=plan.get("relation_requirements", [])
                    relation_missing=_missing_relations(case["question"], answer, result["retrieval"].accepted)
                    mode_stats["relation_total"] += len(relation_specs)
                    mode_stats["relation_covered"] += sum(1 for spec in relation_specs
                        if "relation:"+spec["id"] not in relation_missing)
                    rounds=result.get("generation_rounds", [])
                    round_count = len(rounds) if isinstance(rounds, list) else int(rounds or 0)
                    mode_stats["rounds"].append(round_count)
                    mode_stats["continuations"] += int(round_count > 1)
                    for metric in call_metrics:
                        pms=metric["prompt_ms"]; gms=metric["generation_ms"]
                        pn=metric["prompt_tokens"]; gn=metric["generated_tokens"]
                        mode_stats["prompt_ms"].append(pms); mode_stats["generation_ms"].append(gms)
                        if pms>0 and pn: mode_stats["prompt_tps"].append(pn/(pms/1000))
                        if gms>0 and gn: mode_stats["generation_tps"].append(gn/(gms/1000))
                    rows.append({"mode":mode,"case":case["id"],"retrieval_hit":found,
                        "expected_source":case["source"],"retrieved_sources":[h["source"] for h in hits[:5]],
                        "retrieval_diagnostics":retrieval_diag,
                        "accepted_sources":[h["source"] for h in result["retrieval"].accepted],
                        "answer":answer,"answer_status":status,"answer_reason":result["reason"],
                        "expected_fact":case["answer"],
                        "fact_terms_found":sum(term.lower() in answer.lower() for term in case["terms"]),
                        "fact_terms_total":len(case["terms"]),"grounded_exact_terms":ok,
                        "answer_requirements":requirements,
                        "grounded_requirements":[slot for slot in requirements if slot not in missing],
                        "missing_requirements":sorted(missing),
                        "missing_answer_requirements":sorted(answer_missing),
                        "expected_relation_ids":[spec["id"] for spec in plan.get("relation_requirements", [])],
                        "missing_answer_relations":sorted(relation_missing),
                        "answer_relation_coverage":(1.0 if not relation_specs else
                            round(sum(1 for spec in relation_specs if "relation:"+spec["id"] not in relation_missing)/len(relation_specs),4)),
                        "request_ms":round(total_ms,2),"generation_rounds":round_count,
                        "continuation_count":max(0,round_count-1),
                        "generated_tokens":sum(m["generated_tokens"] for m in call_metrics),
                        "prompt_ms":round(sum(m["prompt_ms"] for m in call_metrics),2),
                        "generation_ms":round(sum(m["generation_ms"] for m in call_metrics),2)})
            store.close()
        metrics_by_mode={}
        for mode,st in stats.items():
            requests=st["requests"]
            metrics_by_mode[mode]={
                "end_to_end_request_ms":{"mean":round(statistics.mean(requests),2),
                    "p50":round(statistics.median(requests),2),
                    "p95":round(sorted(requests)[int(.95*(len(requests)-1))],2)},
                "answer_fact_term_accuracy":round(st["facts"]/len(CASES),4),
                "evidence_requirement_coverage":round(st["plan_covered"]/st["plan_total"],4) if st["plan_total"] else None,
                "answer_requirement_coverage":round(st["answer_covered"]/st["plan_total"],4) if st["plan_total"] else None,
                "answer_requirement_count":st["plan_total"],
                "answer_relation_coverage":round(st["relation_covered"]/st["relation_total"],4) if st["relation_total"] else None,
                "answer_relation_count":st["relation_total"],
                "answer_status_counts":st["statuses"],
                "continuation_rate":round(st["continuations"]/len(CASES),4),
                "mean_generation_rounds":round(statistics.mean(st["rounds"]),2),
                "prompt_processing_ms_mean":round(statistics.mean(st["prompt_ms"]),2) if st["prompt_ms"] else None,
                "prompt_tokens_per_second_mean":round(statistics.mean(st["prompt_tps"]),2) if st["prompt_tps"] else None,
                "generation_ms_mean":round(statistics.mean(st["generation_ms"]),2) if st["generation_ms"] else None,
                "generation_tokens_per_second_mean":round(statistics.mean(st["generation_tps"]),2) if st["generation_tps"] else None}
        mem=proc_memory_mb(proc.pid)
        result={"schema_version":3,"benchmark":"RAG single-pass vs semantic-continuation A/B",
            "platform":platform.platform(),"python":platform.python_version(),"model":Path(args.model).name,
            "threads":args.threads,"context_tokens":args.ctx,"initial_generation_tokens":args.max_tokens,
            "continuation_generation_tokens":min(24,args.max_tokens),"max_continuations":2,
            "documents":len(DOCS),"queries":len(CASES),"server_startup_seconds":round(server_startup,3),
            "retrieval_ms":{"mean":round(statistics.mean(retrieval_ms),3),
                "p50":round(statistics.median(retrieval_ms),3),
                "p95":round(sorted(retrieval_ms)[int(.95*(len(retrieval_ms)-1))],3)},
            "retrieval_source_hit_rate":round(retrieval_hits/len(CASES),4),
            "retrieval_channel_candidate_counts":{key:sum(int(row.get("retrieval_diagnostics",{}).get("channels",{}).get(key,{}).get("candidate_count",0)) for row in rows if row["mode"]=="single") for key in ("fts5_bm25","bigram","exact_match")},
            "metrics_by_mode":metrics_by_mode,"server_memory_mb":mem,"cases":rows,
            "limitations":["Synthetic 8-document corpus and 8 fixed questions; not a human-rated benchmark.",
                "Fact-term match is a strict lexical proxy, not semantic correctness.",
                "GitHub Actions runner metrics are not equivalent to the user's 8GB Intel i5-10310U machine.",
                "Both modes share one server; order alternates to reduce but cannot eliminate prompt-cache effects."]}
        Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:
        proc.terminate()
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        child=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        print(f"LLAMA_SERVER_CHILD_PEAK_RSS_MB={child/1024:.2f}")


if __name__=="__main__": main()
