import asyncio
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, logging


logger = logging.getLogger(__name__)


INDEX_PATH = DATA_DIR / "index.faiss"
CONTENT_PATH = DATA_DIR / "content.json"


_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


async def to_embeddings(texts: list[str]) -> np.ndarray:
    embs: np.ndarray = await asyncio.to_thread(
        _model.encode, 
        texts, 
        convert_to_numpy=True, 
        show_progress_bar=False,
        normalize_embeddings=True
    )
    return embs.astype(np.float32)


async def build_index(content_path: Path = CONTENT_PATH) -> None:
    content = json.loads(content_path.read_text(encoding="utf-8"))
    texts = [it["text"] for it in content]

    embs = await to_embeddings(texts)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, str(INDEX_PATH))
    logger.info('embedding_index создан')


async def load_index(index_path: Path = INDEX_PATH) -> faiss.Index:
    index = await asyncio.to_thread(faiss.read_index, str(index_path))
    return index


async def retrieve(
    index: faiss.Index, 
    content: list[dict[str, Any]], 
    query: str, 
    top_k: int = 5,
    min_score: float | None = 0.3  # 0.35–0.45 — средний порог, 0.5–0.6 — строгий
) -> list[dict[str, Any]]:
    query_emb = await to_embeddings([query])
    scores, ids = await asyncio.to_thread(index.search, query_emb, top_k)
    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], ids[0]):
        if (min_score is not None) and (score < min_score):
            continue
        item = content[idx]
        results.append({"url": item["url"], "text": item["text"]})
    logger.info(f"results: {results}")
    return results

