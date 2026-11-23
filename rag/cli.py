from __future__ import annotations
import argparse, sys, json
from pathlib import Path
from core import Store, extract_file
from answering import answer_with_pipeline, local_llama_generator


def ingest(store: Store, paths: list[Path]) -> int:
    total = 0
    for base in paths:
        if not base.exists():
            print(f'[警告] ファイルがありません: {base}', file=sys.stderr)
            continue
        files = [base] if base.is_file() else sorted(p for p in base.rglob('*') if p.is_file())
        for path in files:
            if path.suffix.lower() not in {'.txt', '.md', '.rst', '.csv', '.log', '.pdf'}:
                continue
            try:
                count = store.add_document(path, extract_file(path))
                total += count
                print(f'[登録] {path.name}: {count} チャンク')
            except Exception as e:
                print(f'[警告] {path}: {e}', file=sys.stderr)
    print(f'完了: {total} チャンクを追加')
    return total


def main():
    ap = argparse.ArgumentParser(description='完全オフラインRAGプロトタイプ')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('ingest', help='文書/フォルダを登録')
    p.add_argument('paths', nargs='+')
    p = sub.add_parser('search', help='文書検索')
    p.add_argument('query')
    p.add_argument('-n', '--limit', type=int, default=5)
    sub.add_parser('chat', help='対話モード')
    args = ap.parse_args()
    config_path = Path(__file__).resolve().parents[1] / 'config' / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    llm = config.get('llm', {})
    generator = None
    if llm.get('enabled'):
        if llm.get('provider') != 'llama.cpp':
            raise SystemExit('未対応LLM providerです。llama.cppのみ利用できます。')
        generator = local_llama_generator(llm.get('endpoint', 'http://127.0.0.1:8080'),
            int(llm.get('max_tokens', 96)), int(llm.get('timeout_seconds', 300)))
    store = Store()
    try:
        if args.cmd == 'ingest':
            ingest(store, [Path(x) for x in args.paths])
        elif args.cmd == 'search':
            for hit in store.search(args.query, max(1, min(args.limit, 20))):
                print(f'[{hit["score"]}] {hit["source"]}\n{hit["text"]}\n')
        else:
            print('Edge RAG / offline prototype. 終了: /exit')
            while True:
                try:
                    q = input('\n質問> ').strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if q.lower() in {'/exit', '/quit'}:
                    break
                if not q:
                    continue
                hits = store.search(q, 12)
                result = answer_with_pipeline(q, hits, generator)
                answer = result["answer"]
                store.log('user', q)
                store.log('assistant', answer)
                print(f'\n[{result["status"]}] {answer}')
    finally:
        store.close()


if __name__ == '__main__':
    main()
