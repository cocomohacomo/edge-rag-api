from __future__ import annotations
import argparse, json, os, platform, random, statistics, tempfile, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.core import Store

TOPICS = [
    ("農業", "農地 土壌 排水 暗渠 農業委員会 現状変更"),
    ("会議", "議事録 決定事項 担当者 期限 合意"),
    ("情報", "データベース 検索 インデックス 文書管理"),
    ("認知", "クラスター フォーカス 状態認知 関心"),
    ("防災", "避難所 洪水 地震 ハザードマップ"),
    ("行政", "条例 申請 審査 通知 手続"),
    ("音声", "文字起こし 話者識別 音声認識"),
    ("機械", "CPU メモリ 性能 ベンチマーク"),
]

def percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, max(0, int((len(ordered)-1)*p)))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", type=int, default=10000)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--output", default="benchmark-results.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "bench.sqlite3"
        store = Store(db_path)
        ingest_start = time.perf_counter()
        expected = []
        for i in range(args.documents):
            topic, terms = TOPICS[i % len(TOPICS)]
            marker = f"管理番号 bench-{i:07d}"
            text = (f"{marker} 分類 {topic} {terms} "
                    f"これは性能評価用の合成日本語文書です。"
                    f"検索対象の主題は{topic}であり、記録内容を再現可能に生成します。\n")
            store.add_document(Path(td) / f"doc-{i:07d}.txt", text)
            expected.append((topic, terms.split()[0], marker))
        ingest_seconds = time.perf_counter() - ingest_start
        store.db.execute("PRAGMA optimize")
        store.db.commit()
        latencies, hits = [], 0
        for q in range(args.queries):
            idx = rng.randrange(args.documents)
            topic, keyword, marker = expected[idx]
            query = f"{topic} {keyword} {marker}"
            start = time.perf_counter()
            results = store.search(query, 5)
            latencies.append((time.perf_counter() - start) * 1000)
            if any(marker in row["text"] for row in results):
                hits += 1
        result = {
            "schema_version": 1,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "documents": args.documents,
            "queries": args.queries,
            "ingest_seconds": round(ingest_seconds, 4),
            "documents_per_second": round(args.documents / max(ingest_seconds, 1e-9), 2),
            "search_ms": {
                "p50": round(statistics.median(latencies), 3),
                "p95": round(percentile(latencies, .95), 3),
                "p99": round(percentile(latencies, .99), 3),
                "mean": round(statistics.mean(latencies), 3),
            },
            "recall_at_5": round(hits / max(args.queries, 1), 4),
            "peak_rss_mb": round(resource_peak_mb(), 2),
            "database_bytes": db_path.stat().st_size,
            "note": "Synthetic benchmark; recall is exact planted-document hit rate, not human relevance."
        }
        store.close()
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

def resource_peak_mb():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024 if os.name != "darwin" else 1024 * 1024)
    except (ImportError, AttributeError):
        return -1

if __name__ == "__main__":
    main()
