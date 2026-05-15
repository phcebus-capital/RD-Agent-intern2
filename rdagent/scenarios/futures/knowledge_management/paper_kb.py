from __future__ import annotations

from pathlib import Path

from rdagent.components.knowledge_management.vector_base import Document, PDVectorBase
from rdagent.log import rdagent_logger as logger

CHUNK_SIZE = 800  # characters per chunk fed into the embedding model


class FuturesPaperKnowledgeBase(PDVectorBase):
    """Vector store for futures-related academic papers.

    Build once from a folder of PDFs, then persist to disk via dump().
    Subsequent runs load from cache instead of re-embedding.
    """

    def build_from_folder(self, folder: str | Path) -> None:
        from rdagent.components.document_reader.document_reader import (
            load_and_process_pdfs_by_langchain,
        )

        folder = Path(folder)
        if not folder.exists():
            logger.warning(f"Paper folder {folder} does not exist — skipping PDF RAG build.")
            return

        docs_dict = load_and_process_pdfs_by_langchain(str(folder))
        if not docs_dict:
            logger.warning(f"No PDFs found in {folder}.")
            return

        logger.info(f"Building paper KB from {len(docs_dict)} PDF(s) in {folder}.")
        for filepath, text in docs_dict.items():
            label = Path(filepath).stem
            for i in range(0, len(text), CHUNK_SIZE):
                chunk = text[i : i + CHUNK_SIZE]
                if chunk.strip():
                    doc = Document(content=chunk, label=label)
                    self.add(doc)  # add() calls create_embedding() internally

        logger.info(f"Paper KB built: {self.shape()[0]} chunks indexed.")

    def retrieve(self, query: str, topk: int = 3) -> str:
        """Return formatted relevant excerpts for a query string, or empty string if none."""
        if self.shape()[0] == 0:
            return ""
        docs, scores = self.search(query, topk_k=topk, similarity_threshold=0.3)
        if not docs:
            return ""
        lines = ["### Relevant Paper Excerpts"]
        for doc, score in zip(docs, scores):
            lines.append(f"[{doc.label} | sim={score:.2f}]\n{doc.content}")
        return "\n\n".join(lines)
