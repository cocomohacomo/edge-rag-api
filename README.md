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
