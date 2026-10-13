import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class ChromaKnowledgeStore:
    """
    Persistent ChromaDB vector store dedicated to teacher knowledge index.
    Does NOT act as authoritative exam source of truth (that remains in the primary structured DB).
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_persist = os.path.join(base_dir, "data", "chroma")
        
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", default_persist)
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "teacher_knowledge")
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Teacher exam knowledge base: question papers, model answers, and rubrics"}
        )
        logger.info(f"[CHROMA] Initialized persistent ChromaDB at '{self.persist_directory}' with collection '{self.collection_name}'.")

    def delete_test_chunks(self, test_id: str) -> int:
        """
        Removes all documents and embeddings associated with a given test_id.
        Ensures idempotency and clean re-ingestion.
        """
        try:
            # Query existing IDs for this test_id
            existing = self.collection.get(where={"test_id": test_id})
            existing_ids = existing.get("ids", [])
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                logger.info(f"[CHROMA] Deleted {len(existing_ids)} existing chunks for test_id='{test_id}'.")
                return len(existing_ids)
            return 0
        except Exception as e:
            logger.error(f"[CHROMA] Failed deleting chunks for test_id='{test_id}': {e}")
            raise

    def upsert_chunks(
        self,
        test_id: str,
        chunk_documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> int:
        """
        Inserts or replaces semantic chunks with pre-computed embeddings and metadata.
        """
        if not chunk_documents:
            logger.warning(f"[CHROMA] No documents provided to upsert for test_id='{test_id}'.")
            return 0

        if len(chunk_documents) != len(embeddings) or len(chunk_documents) != len(metadatas):
            raise ValueError(
                f"Count mismatch: documents={len(chunk_documents)}, embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )

        # Generate deterministic IDs if not explicitly provided
        if ids is None:
            ids = [
                f"{test_id}_{meta.get('question_id', 'gen')}_{idx}"
                for idx, meta in enumerate(metadatas)
            ]

        # ChromaDB metadata values must be str, int, float, or bool
        cleaned_metadatas = []
        for meta in metadatas:
            clean_m = {}
            for k, v in meta.items():
                if v is None:
                    clean_m[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    clean_m[k] = v
                else:
                    clean_m[k] = str(v)
            cleaned_metadatas.append(clean_m)

        try:
            self.collection.upsert(
                ids=ids,
                documents=chunk_documents,
                embeddings=embeddings,
                metadatas=cleaned_metadatas
            )
            logger.info(f"[CHROMA] Collection: {self.collection_name} | Stored/Upserted {len(ids)} documents for test_id='{test_id}'.")
            return len(ids)
        except Exception as e:
            logger.error(f"[CHROMA] Upsert error: {e}")
            raise RuntimeError(f"ChromaDB upsert failed: {str(e)}")

    def get_test_chunks(self, test_id: str) -> Dict[str, Any]:
        """Retrieves stored chunks and metadata for a specific test_id."""
        return self.collection.get(where={"test_id": test_id})

    def query_test_knowledge(self, test_id: str, query_embedding: List[float], top_k: int = 2) -> List[Dict[str, Any]]:
        """Queries the vector store for semantic matches strictly within a specific test_id."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"test_id": test_id}
            )
            
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            
            matches = []
            for d, m, dist in zip(docs, metas, dists):
                matches.append({
                    "document": d,
                    "metadata": m,
                    "distance": dist
                })
            return matches
        except Exception as e:
            logger.error(f"[CHROMA] Query failed for test_id='{test_id}': {e}")
            return []

    def inspect_collection(self) -> Dict[str, Any]:
        """Debug helper to inspect the knowledge base contents without exposing evaluation/search APIs."""
        count = self.collection.count()
        sample = self.collection.get(limit=100)
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "total_documents": count,
            "sample_ids": sample.get("ids", []),
            "sample_metadatas": sample.get("metadatas", []),
            "sample_documents": sample.get("documents", [])[:5] if sample.get("documents") else []
        }

# Singleton instance
knowledge_store = ChromaKnowledgeStore()
