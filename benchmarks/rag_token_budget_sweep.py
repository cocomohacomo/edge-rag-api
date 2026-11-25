from __future__ import annotations
import argparse, json, statistics, subprocess, time, tempfile, platform, resource, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.core import Store
from rag.answering import answer_with_pipeline, _missing_slots, _prompt_question, _is_continuation_prompt
from rag.retrieval_v2 import search_with_diagnostics
from rag_e2e_benchmark import CASES, DOCS, get_json

def main():
    ap=argparse.ArgumentParser(description="Sweep max generation token budgets for semantic-continuation RAG")
    ap.add_argument("--model", required=True)
    ap.add_argument("--server", default="llama.cpp/build/bin/llama-server")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--budgets", default="10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--max-continuations", type=int, default=2)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--output", default="token-budget-sweep.json")
    args=ap.parse_args()
    budgets=sorted(set(int(x.strip()) for x in args.budgets.split(",") if x.strip()))
    if not budgets or any(x < 1 or x > 100 for x in budgets):
        raise ValueError("budgets must be integers in range 1..100")
    server=[args.server,"-m",args.model,"--host","127.0.0.1","--port",str(args.port),"-t",str(args.threads),"-c",str(args.ctx),"-ngl","0"]
    proc=subprocess.Popen(server,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    base=f"http://127.0.0.1:{args.port}"
    try:
        ready=False
        for _ in range(180):
            if proc.poll() is not None: raise RuntimeError(f"llama-server exited: {proc.returncode}")
            try: get_json(base+"/health",timeout=2); ready=True; break
            except Exception: time.sleep(1)
        if not ready: raise TimeoutError("llama-server readiness timeout")
        results=[]
        with tempfile.TemporaryDirectory() as td:
            store=Store(Path(td)/"sweep.sqlite3")
            for name,text in DOCS: store.add_document(Path(td)/name,text)
            for budget_cap in budgets:
                rows=[]; total_ms=[]; gen_ms=[]; prompt_ms=[]; tokens=[]; continuations=0
                facts=0; evidence_cov=0; answer_cov=0; req_total=0; statuses={}
                for case in CASES:
                    hits,_=search_with_diagnostics(store,case["question"],12)
                    call_metrics=[]
                    def generate(prompt):
                        q=_prompt_question(prompt)
                        budget=min(budget_cap,24) if _is_continuation_prompt(prompt) else budget_cap
                        response=get_json(base+"/completion",{"prompt":prompt,"n_predict":budget,"temperature":0.0,
                            "top_k":1,"top_p":1.0,"seed":42,"cache_prompt":True},timeout=300)
                        timing=response.get("timings",{})
                        call_metrics.append({"prompt_ms":float(timing.get("prompt_ms",0)),
                            "generation_ms":float(timing.get("predicted_ms",0)),
                            "prompt_tokens":int(timing.get("prompt_n",0)),
                            "generated_tokens":int(timing.get("predicted_n",0)),"n_predict":budget,
                            "stop":response.get("stop"),"truncated":response.get("truncated")})
                        return {"content":response.get("content") or "", "stop":response.get("stop"),
                            "stopped_eos":response.get("stopped_eos"),"stopped_word":response.get("stopped_word"),
                            "stopped_limit":response.get("stopped_limit"),"stop_type":response.get("stop_type"),
                            "truncated":response.get("truncated"),"tokens_predicted":int(timing.get("predicted_n",0)),
                            "n_predict":budget}
                    started=time.perf_counter()
                    result=answer_with_pipeline(case["question"],hits,generate,retry=False,
                        continuation=True,max_continuations=args.max_continuations,
                        prefer_evidence_assembly=False,force_generation=True)
                    elapsed=(time.perf_counter()-started)*1000
                    answer=result["answer"]; total_ms.append(elapsed)
                    statuses[result["status"]]=statuses.get(result["status"],0)+1
                    facts+=int(all(term.lower() in answer.lower() for term in case["terms"]))
                    plan=result.get("answer_plan",{}); req=plan.get("requirements",[])
                    missing=set(plan.get("missing_requirements",[]))
                    ans_missing=set(_missing_slots(case["question"],answer,result["retrieval"].accepted))
                    req_total+=len(req); evidence_cov+=sum(1 for slot in req if slot not in missing)
                    answer_cov+=sum(1 for slot in req if slot not in ans_missing)
                    rounds=result.get("generation_rounds",[])
                    rc=len(rounds) if isinstance(rounds,list) else int(rounds or 0)
                    continuations+=int(rc>1)
                    gen_ms.extend(m["generation_ms"] for m in call_metrics)
                    prompt_ms.extend(m["prompt_ms"] for m in call_metrics)
                    tokens.extend(m["generated_tokens"] for m in call_metrics)
                    rows.append({"case":case["id"],"status":result["status"],"answer":answer,
                        "fact_terms_found":sum(term.lower() in answer.lower() for term in case["terms"]),
                        "fact_terms_total":len(case["terms"]),"requirements":req,
                        "missing_requirements":result.get("missing_slots",[]),"request_ms":round(elapsed,2),
                        "rounds":rc,"generation_calls":len(call_metrics),"continuation_count":max(0,rc-1),
                        "generated_tokens":sum(m["generated_tokens"] for m in call_metrics),
                        "generation_ms":round(sum(m["generation_ms"] for m in call_metrics),2),
                        "prompt_ms":round(sum(m["prompt_ms"] for m in call_metrics),2)})
                results.append({"max_token_budget":budget_cap,"case_count":len(CASES),
                    "mean_response_ms":round(statistics.mean(total_ms),2),
                    "p50_response_ms":round(statistics.median(total_ms),2),
                    "p95_response_ms_nearest_rank":round(sorted(total_ms)[max(0,math.ceil(.95*len(total_ms))-1)],2),
                    "max_response_ms":round(max(total_ms),2),
                    "fact_term_accuracy":round(facts/len(CASES),4),
                    "evidence_requirement_coverage":round(evidence_cov/req_total,4) if req_total else None,
                    "answer_requirement_coverage":round(answer_cov/req_total,4) if req_total else None,
                    "continuation_rate":round(continuations/len(CASES),4),
                    "generation_invoked_cases":sum(1 for row in rows if row["generation_calls"]>0),
                    "generation_invocation_rate":round(sum(1 for row in rows if row["generation_calls"]>0)/len(CASES),4),
                    "mean_generated_tokens":round(statistics.mean(tokens),2) if tokens else 0,
                    "mean_generation_ms":round(statistics.mean(gen_ms),2) if gen_ms else None,
                    "mean_prompt_ms":round(statistics.mean(prompt_ms),2) if prompt_ms else None,
                    "status_counts":statuses,"cases":rows})
                print(f"budget={budget_cap}: mean={results[-1]['mean_response_ms']}ms facts={results[-1]['fact_term_accuracy']} continuations={results[-1]['continuation_rate']}",flush=True)
            store.close()
        output={"schema_version":1,"benchmark":"semantic-continuation token-budget sweep",
            "platform":platform.platform(),"python":platform.python_version(),"model":Path(args.model).name,
            "threads":args.threads,"context_tokens":args.ctx,"budgets":budgets,
            "max_continuations":args.max_continuations,"documents":len(DOCS),"queries_per_budget":len(CASES),
            "metrics_by_budget":results,
            "selection_rule":"Choose the smallest token budget that reaches the best fact-term accuracy and answer-requirement coverage; use response latency and continuation rate as tie-breakers.",
            "limitations":["Synthetic corpus and fixed 8-query set; not human-rated semantic quality.",
            "Fact-term matching is a lexical proxy only.","GitHub Actions runner is not equivalent to the target Windows 8GB machine.",
            "Budgets are tested in ascending order on one server; prompt-cache warming may affect latency."]}
        Path(args.output).write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(output,ensure_ascii=False,indent=2))
    finally:
        proc.terminate()
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        print(f"LLAMA_SERVER_CHILD_PEAK_RSS_MB={resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024:.2f}")

if __name__=="__main__": main()
