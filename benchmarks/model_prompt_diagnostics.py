from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def request(url: str, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--output", default="model-prompt-diagnostics.json")
    ap.add_argument("--log", default="model-prompt-diagnostics-server.log")
    args = ap.parse_args()

    cmd = [args.server, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port),
           "-t", str(args.threads), "-c", str(args.ctx), "-ngl", "0"]
    with open(args.log, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{args.port}"
        try:
            ready = False
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited early with code {proc.returncode}")
                try:
                    request(base + "/health", timeout=2)
                    ready = True
                    break
                except Exception:
                    time.sleep(1)
            if not ready:
                raise TimeoutError("server readiness timeout")

            props = {}
            try:
                props = request(base + "/props", timeout=5)
            except Exception as exc:
                props = {"error": type(exc).__name__}

            question = "会議の担当者は誰ですか。根拠: 担当者は佐藤です。"
            prompts = {
                "plain_completion": f"質問: {question}\n回答:",
                "instruction_completion": (
                    "あなたは根拠に忠実な日本語回答器です。根拠に書かれた事実だけを使い、"
                    "回答本文だけを簡潔に出力してください。\n"
                    f"質問本文\n{question}\n参照資料\n担当者は佐藤です。\n回答本文\n"
                ),
                "chatml_user_assistant": (
                    f"<|im_start|>user\n{question}\n根拠: 担当者は佐藤です。"
                    "<|im_end|>\n<|im_start|>assistant\n"
                ),
                "llama_inst": f"[INST] {question} 根拠: 担当者は佐藤です。 [/INST]",
            }
            results = {}
            # Exercise the model/server-provided chat template separately from raw completion.
            try:
                chat = request(base + "/v1/chat/completions", {
                    "model": Path(args.model).name,
                    "messages": [
                        {"role": "system", "content": "根拠に忠実な日本語回答器です。根拠にある事実だけを回答してください。"},
                        {"role": "user", "content": question + "\n根拠: 担当者は佐藤です。"},
                    ],
                    "temperature": 0.0, "max_tokens": 48, "seed": 42,
                })
                results["server_native_chat_template"] = {
                    "raw_response": chat,
                    "raw_output": (((chat.get("choices") or [{}])[0].get("message") or {}).get("content", "")),
                    "finish_reason": (chat.get("choices") or [{}])[0].get("finish_reason"),
                }
            except Exception as exc:
                results["server_native_chat_template"] = {
                    "error": type(exc).__name__, "message": str(exc)
                }
            for name, prompt in prompts.items():
                try:
                    result = request(base + "/completion", {
                        "prompt": prompt, "n_predict": 48, "temperature": 0.0,
                        "top_k": 1, "top_p": 1.0, "seed": 42, "cache_prompt": False,
                    })
                    results[name] = {
                        "prompt": prompt,
                        "raw_response": result,
                        "raw_output": result.get("content", result.get("raw", "")),
                        "stop": result.get("stop"),
                        "stop_type": result.get("stop_type"),
                        "stopped_eos": result.get("stopped_eos"),
                        "stopped_word": result.get("stopped_word"),
                        "stopped_limit": result.get("stopped_limit"),
                        "tokens_predicted": result.get("tokens_predicted"),
                        "timings": result.get("timings"),
                    }
                except Exception as exc:
                    results[name] = {"error": type(exc).__name__, "message": str(exc)}

            report = {
                "model": Path(args.model).name,
                "threads": args.threads,
                "context": args.ctx,
                "server_props": props,
                "prompt_variants": results,
                "interpretation_note": (
                    "Raw completion outputs are preserved. A format is not considered valid "
                    "merely because it returns text; verify answer grounding and absence of prompt echoes."
                ),
            }
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
