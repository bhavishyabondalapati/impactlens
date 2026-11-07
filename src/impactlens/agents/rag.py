"""SOP retrieval: download -> extract text -> chunk -> embed locally -> FAISS index -> search.

- Embeddings: sentence-transformers all-MiniLM-L6-v2 (384-d, runs on the laptop, no API cost).
- Index: FAISS IndexFlatIP over L2-normalised vectors, so inner product = cosine similarity.
  "Flat" means exact search; with a few hundred chunks that is instant and needs no training.
- Chunks: ~220 words with 40 words of overlap, each tagged with its source and page, so the
  Reporter can cite "[S3]" and a reader can find the passage.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from impactlens.agents.sources import SOURCES, Source, sources_markdown

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
USER_AGENT = "impactlens/0.1 (open-source research project)"
SOP_DIR = Path("data/raw/sops")
INDEX_DIR = Path("data/processed/rag")


@dataclass
class Chunk:
    chunk_id: str  # e.g. "fao:p12:3"
    source: str  # Source.sid
    title: str
    page: int | None  # PDF page (1-based); None for web pages
    text: str


# ------------------------------------------------------------------------------- download


def download(sources=SOURCES, sop_dir: Path = SOP_DIR, log=print) -> None:
    """Fetch any missing document and (re)write sources.md next to them."""
    sop_dir.mkdir(parents=True, exist_ok=True)
    for s in sources:
        path = sop_dir / s.file
        if path.exists() and path.stat().st_size > 0:
            continue
        log(f"downloading {s.url}")
        req = urllib.request.Request(s.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    (sop_dir / "sources.md").write_text(sources_markdown())


# ------------------------------------------------------------------------------- extract


class _TextOnly(HTMLParser):
    """Visible text of a web page, without scripts, styles, menus, headers and footers."""

    SKIP = {"script", "style", "nav", "header", "footer", "noscript"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        self._skip += tag in self.SKIP

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def extract_pages(path: Path) -> list[tuple[int | None, str]]:
    """[(page number or None, text)] for a PDF or an HTML page."""
    if path.suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    parser = _TextOnly()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return [(None, " ".join(parser.parts))]


def chunk_words(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    if words == [""]:
        return []
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def build_chunks(sources=SOURCES, sop_dir: Path = SOP_DIR, min_words: int = 40) -> list[Chunk]:
    chunks = []
    for s in sources:
        for page, text in extract_pages(sop_dir / s.file):
            for k, piece in enumerate(chunk_words(text)):
                if len(piece.split()) >= min_words:  # skip near-empty pages (covers, blanks)
                    pid = f"p{page}" if page else "web"
                    chunks.append(Chunk(f"{s.sid}:{pid}:{k}", s.sid, s.title, page, piece))
    return chunks


# ------------------------------------------------------------------------- embed + index


def _faiss():
    """Import faiss so it can live in the same process as torch.

    On macOS the faiss-cpu and torch wheels each bundle their own copy of the OpenMP runtime
    (libomp.dylib). Whichever initialises second aborts the process ("OMP: Error #15") unless
    KMP_DUPLICATE_LIB_OK=TRUE is set. That flag is officially "unsafe" because two runtimes
    can disagree, so we (1) load torch first and set the flag only here, right before faiss
    loads (faiss first then torch segfaults even with the flag), (2) run faiss on one
    thread (a few hundred vectors need no parallelism), and (3) test that faiss returns the
    same neighbours as a brute-force numpy search after torch has run
    (tests/test_agents.py::test_faiss_matches_brute_force_next_to_torch).
    """
    import importlib
    import os

    try:
        import torch  # noqa: F401  torch's runtime must come first: the other order segfaults
    except ImportError:
        pass  # the AWS Lambda image has no torch (ONNX embedder), so there is no clash there

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    faiss = importlib.import_module("faiss")
    faiss.omp_set_num_threads(1)
    return faiss


class Embedder:
    """Thin wrapper so tests can swap in a fake (no model download)."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def __call__(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def build_index(chunks: list[Chunk], embed, out_dir: Path = INDEX_DIR) -> Path:
    faiss = _faiss()
    vecs = embed([c.text for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "sops.faiss"))
    with (out_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    return out_dir


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR, embed=None):
        faiss = _faiss()
        self.index = faiss.read_index(str(index_dir / "sops.faiss"))
        lines = (index_dir / "chunks.jsonl").read_text().splitlines()
        self.chunks = [Chunk(**json.loads(line)) for line in lines]
        self.embed = embed or Embedder()

    def search(self, query: str, k: int = 4) -> list[tuple[float, Chunk]]:
        scores, ids = self.index.search(self.embed([query]), k)
        pairs = zip(scores[0], ids[0], strict=True)
        return [(float(s), self.chunks[i]) for s, i in pairs if i >= 0]


def source_by_id(sid: str) -> Source:
    return next(s for s in SOURCES if s.sid == sid)
