# edge-rag-api: Windows Offline RAG Prototype

初版は **ネットワーク接続なし** で文書登録・日本語検索・根拠抜粋チャットを動かす最小プロトタイプです。LLM/Embeddingのモデルファイルはまだ同梱していません。LLM未設定時は検索結果を根拠として表示し、生成AIによる統合回答を装いません。

## 対象
- Windows 11 / Python 3.10+
- SQLite FTS5（標準SQLite）
- TXT / Markdown / CSV / RST / LOG
- PDFはローカルに `pypdf` を追加した場合のみ
- 外部API・DNS・モデル自動ダウンロードなし

## 起動
```bat
launcher\setup_check.bat
launcher\ingest.bat "D:\\docs"
launcher\start.bat
```
または `python rag\cli.py ingest <file-or-folder>` / `python rag\cli.py search "検索語"`。

## 重要な制約
- 初版はBM25/日本語文字bigramによる語彙検索です。ONNX埋め込み・ベクトル検索は未実装。
- LLMは未接続です。回答は検索根拠の抜粋であり、要約・推論生成ではありません。
- スキャンPDFのOCRは未対応。
- `runtime/python`、LLM、Embedding、pypdf等の依存物は配布用USBパッケージ工程で別PCから収集します。現時点のリポジトリはソースプロトタイプです。
- 1MはLLMの単一プロンプト長ではなく、RAG保管・検索対象容量の目標として扱います。

## テスト
```bash
python -m unittest discover -s tests -v
```

## 完全オフライン設計
本アプリケーションコードから外部通信は行いません。Python実行環境やOS側の通信を含めた「外部通信ゼロ」の証明は、対象端末でファイアウォール規則・パケットキャプチャ等を用いて別途確認してください。


## 回答制御パイプライン
`rag.answering` は追加モデル不要の軽量な回答制御層です。

- Retrieval Gate: 既存のFTS5/BM25順位に質問語彙の被覆率を加え、低関連チャンクを除外
- Evidence-first Prompt: 根拠限定、直接回答、推測禁止、根拠不足時の棄権を明示
- Generation Validator: 空回答、根拠語彙の不足、根拠にない数値を検出
- Retry: 不合格時に制約を強めて最大1回だけ再生成
- Extractive Fallback: 再生成も不合格なら質問語との一致度が高い原文文を出典付きで返す

利用例:

```python
from rag.answering import answer_with_pipeline
hits = store.search(question, 12)
result = answer_with_pipeline(question, hits, generator=my_local_llm)
print(result["answer"], result["status"])
```

`generator` はプロンプト文字列を受け取り回答文字列を返す関数です。LLM未接続時は省略でき、その場合も根拠抽出型回答として動作します。検証器は軽量な決定的ガードであり、意味的正しさを完全保証するものではありません。


ローカルRWKV/llama.cppを接続する場合は、llama-serverをloopbackで起動したうえで `config/config.json` の `llm.enabled` を `true` にします。接続先は `127.0.0.1` / `localhost` / `::1` に限定され、外部APIには送信しません。未起動・生成失敗時は検証後に抽出型フォールバックへ移行します。
