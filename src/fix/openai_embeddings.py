"""LlamaIndex adapter for a school's OpenAI-compatible embeddings endpoint."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import PrivateAttr
from llama_index.core.base.embeddings.base import BaseEmbedding


class OpenAICompatibleEmbedding(BaseEmbedding):
    """Use the same remote model for document chunks and retrieval queries.

    The API key is private so LlamaIndex's serialized model metadata omits it.
    Each request closes its client; HTTPX honors NO_PROXY from the environment.
    """

    _api_key: str = PrivateAttr()
    _base_url: str = PrivateAttr()
    _timeout: float = PrivateAttr()
    _query_instruction: str | None = PrivateAttr()

    def __init__(self, *, model_name: str, base_url: str, api_key: str,
                 timeout: float = 60, query_instruction: str | None = None,
                 **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._query_instruction = query_instruction

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            with OpenAI(api_key=self._api_key, base_url=self._base_url,
                        timeout=self._timeout, max_retries=2) as client:
                response = client.embeddings.create(
                    model=self.model_name, input=texts, encoding_format="float",
                )
        except APIConnectionError as exc:
            cause = exc.__cause__
            while cause is not None:
                if isinstance(cause, httpx.ProxyError):
                    raise RuntimeError(
                        "Embedding proxy connection failed. Check NO_PROXY for the "
                        "school hostname and the required campus network/VPN."
                    ) from exc
                cause = cause.__cause__
            raise RuntimeError(
                "Cannot connect to the embedding service. Check its URL, campus "
                "network/VPN, proxy settings and certificate configuration."
            ) from exc
        except APIStatusError as exc:
            raise RuntimeError(
                f"Embedding service returned HTTP {exc.status_code}. "
                "Check API key/model permissions, endpoint and request limits."
            ) from exc

        # Servers may return batch results out of order. Never attach a vector
        # to the wrong document, or silently accept missing/duplicate results.
        data = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in data] != list(range(len(texts))):
            raise ValueError("Embedding response has missing or duplicate indices.")
        vectors = [item.embedding for item in data]
        dimension = len(vectors[0])
        if dimension == 0 or any(len(v) != dimension for v in vectors):
            raise ValueError("Embedding response has empty or inconsistent dimensions.")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("Embedding response contains non-finite values.")
        return vectors

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def _get_query_embedding(self, query: str) -> list[float]:
        if self._query_instruction:
            query = f"Instruct: {self._query_instruction}\nQuery:{query}"
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed, texts)
