import json, os, re, subprocess, sys, time
from pathlib import Path

model = Path(os.environ["MODEL_PATH"])
binary = Path(os.environ["LLAMA_CLI"])
prompt = ("User: 日本語で、オフライン環境における検索拡張生成の利点を"
          "具体例を交えて説明してください。\nAssistant:")
threads = [1, 2, 4]
results = []
for n_threads in threads:
    cmd = [str(binary), "-m", str(model), "-t", str(n_threads),
           "-c", "2048", "-n", "128", "-p", prompt,
           "--temp", "0", "--seed", "1", "--no-display-prompt",
           "--no-warmup", "--simple-io", "--verbose"]
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=900)
    elapsed = time.perf_counter() - start
    log = proc.stdout + "\n" + proc.stderr
    if proc.returncode:
        print(log[-12000:])
        raise SystemExit(f"llama-cli failed: exit={proc.returncode}")
    prompt_match = re.search(r"prompt eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens.*?([\d.]+) tokens per second", log, re.S)
    eval_match = re.search(r"eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) runs.*?([\d.]+) tokens per second", log, re.S)
    results.append({
        "threads": n_threads, "wall_seconds": round(elapsed, 3),
        "prompt_eval": ({"milliseconds": float(prompt_match.group(1)),
                         "tokens": int(prompt_match.group(2)),
                         "tokens_per_second": float(prompt_match.group(3))}
                        if prompt_match else None),
        "generation": ({"milliseconds": float(eval_match.group(1)),
                        "tokens": int(eval_match.group(2)),
                        "tokens_per_second": float(eval_match.group(3))}
                       if eval_match else None),
        "returncode": proc.returncode,
        "log_tail": log[-2500:]
    })
    print(json.dumps(results[-1], ensure_ascii=False))
report = {"model": "shoumenchougou/RWKV7-G1l-1.5B-GGUF",
          "quantization": "Q4_K_M", "runner": os.getenv("RUNNER_NAME", "GitHub-hosted"),
          "results": results}
Path("rwkv-inference-results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
if not all(x["generation"] for x in results):
    raise SystemExit("Could not parse generation tokens/sec from llama.cpp logs")
