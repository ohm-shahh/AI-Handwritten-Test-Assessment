import os
import logging
from typing import List, Tuple
import httpx
import ollama
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class OllamaEmbeddingService(Embeddings):
    """
    Dedicated embedding service utilizing qwen3-embedding:4b via Ollama.
    Implements LangChain Embeddings interface for seamless integration with SemanticChunker.
    """
    def __init__(self, model_name: str = "qwen3-embedding:4b", base_url: str = "http://127.0.0.1:11434"):
        self.model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", model_name)
        self.base_url = os.getenv("OLLAMA_BASE_URL", base_url)
        self.client = ollama.Client(host=self.base_url, timeout=60.0)

    def is_available(self) -> Tuple[bool, str]:
        """
        Verifies that the Ollama host is accessible and that the requested
        embedding model (qwen3-embedding:4b) is installed and available.
        """
        try:
            models_info = self.client.list()
            # Handle both dict and object response formats from ollama python SDK
            models = models_info.get("models", []) if isinstance(models_info, dict) else getattr(models_info, "models", [])
            
            available_names = []
            for m in models:
                name = m.get("name") if isinstance(m, dict) else getattr(m, "model", getattr(m, "name", ""))
                if name:
                    available_names.append(name.lower())

            model_base = self.model_name.lower().split(":")[0]
            matched = any(self.model_name.lower() in m or model_base in m for m in available_names)

            if matched:
                return True, f"Ollama model '{self.model_name}' is ready."
            else:
                return False, f"Model '{self.model_name}' not found in Ollama. Available models: {', '.join(available_names)}"
        except Exception as e:
            logger.error(f"[EMBED] Ollama connectivity check failed: {e}")
            return False, f"Ollama service unreachable at {self.base_url}: {str(e)}"

    def embed_query(self, text: str) -> List[float]:
        """Generates embedding for a single query text."""
        return self.get_embedding(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of documents."""
        return self.get_embeddings(texts)

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding vector for a single text using qwen3-embedding:4b.
        """
        if not text or not text.strip():
            # Return empty or dummy vector if string is completely empty
            text = " "

        clean_text = text.strip()
        try:
            res = self.client.embeddings(model=self.model_name, prompt=clean_text)
            vector = res.get("embedding", []) if isinstance(res, dict) else getattr(res, "embedding", [])
            if not vector:
                raise ValueError(f"Ollama returned empty embedding for model '{self.model_name}'.")
            return vector
        except Exception as e:
            logger.error(f"[EMBED] Failed to generate embedding with '{self.model_name}': {e}")
            raise RuntimeError(f"Embedding generation failed for model '{self.model_name}': {str(e)}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a list of texts sequentially.
        """
        if not texts:
            return []

        logger.info(f"[EMBED] Model: {self.model_name} | Generating embeddings for {len(texts)} chunks...")
        embeddings = []
        for idx, text in enumerate(texts):
            vec = self.get_embedding(text)
            embeddings.append(vec)

        logger.info(f"[EMBED] Successfully generated {len(embeddings)} embeddings ({len(embeddings[0]) if embeddings else 0} dimensions).")
        return embeddings

# Singleton instance
embedding_service = OllamaEmbeddingService()
