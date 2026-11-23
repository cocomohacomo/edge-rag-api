from __future__ import annotations
import hashlib, re, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / 'storage' / 'database' / 'rag.sqlite3'


def japanese_terms(text: str) -> list[str]:
    """Japanese bigrams plus Latin/numeric tokens for FTS5."""
    text = re.sub(r'\\s+', ' ', text.lower())
    terms: list[str] = []
    for token in re.findall(r'[ぁ-んァ-ヶ一-龠ー]+|[a-z0-9_]+', text):
        if re.fullmatch(r'[ぁ-んァ-ヶ一-龠ー]+', token):
            terms.extend([token] if len(token) == 1 else
                         [token[i:i+2] for i in range(len(token)-1)])
        else:
            terms.append(token)
    return terms


def chunks(text: str, size: int = 420, overlap: int = 70) -> list[str]:
    text = re.sub(r'\\r\\n?', '\\n', text).strip()
    if not text:
        return []
    out, pos = [], 0
    while pos < len(text):
        end = min(len(text), pos + size)
        if end < len(text):
            cut = max(text.rfind('\\n', pos, end), text.rfind('。', pos, end),
                      text.rfind(' ', pos, end))
            if cut > pos + size // 2:
                end = cut + 1
        part = text[pos:end].strip()
        if part:
            out.append(part)
        if end >= len(text):
            break
        pos = max(pos + 1, end - overlap)
    return out


class Store:
    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self._schema()

    def _schema(self):
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS documents(
          id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, title TEXT NOT NULL,
          sha256 TEXT NOT NULL, added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS chunks(
          id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          ordinal INTEGER NOT NULL, content TEXT NOT NULL, terms TEXT NOT NULL,
          UNIQUE(document_id, ordinal));
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(terms, content='chunks', content_rowid='id');
        CREATE TABLE IF NOT EXISTS conversations(
          id INTEGER PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        ''')
        self.db.executescript('''
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
          INSERT INTO chunks_fts(rowid, terms) VALUES (new.id, new.terms);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
          INSERT INTO chunks_fts(chunks_fts, rowid, terms) VALUES('delete', old.id, old.terms);
        END;
        ''')
        self.db.commit()

    def add_document(self, path: Path, text: str) -> int:
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        key = str(path.resolve())
        old = self.db.execute('SELECT id, sha256 FROM documents WHERE path=?', (key,)).fetchone()
        if old and old['sha256'] == digest:
            return 0
        if old:
            self.db.execute('DELETE FROM documents WHERE id=?', (old['id'],))
        cur = self.db.execute('INSERT INTO documents(path,title,sha256) VALUES(?,?,?)',
                              (key, path.name, digest))
        doc_id = cur.lastrowid
        rows = [(doc_id, i, part, ' '.join(japanese_terms(part)))
                for i, part in enumerate(chunks(text))]
        self.db.executemany('INSERT INTO chunks(document_id,ordinal,content,terms) VALUES(?,?,?,?)', rows)
        self.db.commit()
        return len(rows)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        terms = japanese_terms(query)
        if not terms:
            return []
        expr = ' OR '.join('"' + t.replace('"', '""') + '"'
                           for t in dict.fromkeys(terms))
        try:
            rows = self.db.execute('''SELECT c.id,c.content,d.title,d.path,bm25(chunks_fts) AS rank
              FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
              JOIN documents d ON d.id=c.document_id WHERE chunks_fts MATCH ?
              ORDER BY rank LIMIT ?''', (expr, max(limit * 4, limit))).fetchall()
        except sqlite3.OperationalError:
            rows = []
        qset = set(terms)
        scored = []
        for row in rows:
            cterms = set(japanese_terms(row['content']))
            overlap = len(qset & cterms) / max(1, len(qset))
            fts_score = 1.0 / (1.0 + max(0.0, float(row['rank'])))
            scored.append((0.65 * overlap + 0.35 * fts_score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{'score': round(s, 4), 'text': r['content'],
                 'source': r['title'], 'path': r['path']} for s, r in scored[:limit]]

    def log(self, role: str, content: str):
        self.db.execute('INSERT INTO conversations(role,content) VALUES(?,?)',
                        (role, content))
        self.db.commit()

    def history(self, limit: int = 20):
        return self.db.execute(
            'SELECT role,content,created_at FROM conversations ORDER BY id DESC LIMIT ?',
            (limit,)).fetchall()[::-1]

    def close(self):
        self.db.close()


def extract_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {'.txt', '.md', '.rst', '.csv', '.log'}:
        return path.read_text(encoding='utf-8-sig', errors='replace')
    if suffix == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError('PDF読込にはローカル環境に pypdf が必要です。') from e
        reader = PdfReader(str(path))
        return '\\n'.join(page.extract_text() or '' for page in reader.pages)
    raise ValueError(f'未対応ファイル形式: {suffix}')


def answer_from_context(question: str, hits: list[dict]) -> str:
    if not hits:
        return '関連する文書は見つかりませんでした。文書を登録するか、質問の表現を変えてください。'
    lines = ['検索した文書から、関連箇所を抜粋します。', '']
    for i, hit in enumerate(hits[:4], 1):
        excerpt = hit['text'].strip()
        if len(excerpt) > 700:
            excerpt = excerpt[:700] + '…'
        lines.extend([f'[{i}] {hit["source"]} (関連度 {hit["score"]})', excerpt, ''])
    lines.append('※ローカル検索による根拠抜粋です。LLM未設定のため、複数箇所の統合・推論は行っていません。')
    return '\\n'.join(lines)
