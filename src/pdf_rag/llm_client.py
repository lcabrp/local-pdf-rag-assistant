"""Local-first LLM client (Ollama primary -> OpenAI cloud fallback).

Why this module exists:
-----------------------
`local-pdf-rag-assistant` is designed to run as much of the RAG pipeline as
possible on local hardware. While heavy coding/reasoning models require large
server GPUs, answering focused RAG questions from 3 pre-selected parent
passages (~1,200 tokens) fits comfortably inside a small local model.

Design decisions:
1. **Local-first, Cloud fallback:** Every query tries local Ollama first and only
   calls OpenAI (`gpt-6-luna`) if Ollama is unavailable, errors, returns
   malformed JSON, or spills layers onto the CPU.
2. **Native Ollama `/api/chat` instead of `/v1` shim:**
   - Supports `"think": false` so reasoning models (like Gemma 4) don't burn
     time/tokens on hidden chain-of-thought when a concise JSON reply is wanted.
   - Supports `"format": "json"` for grammar-constrained JSON output.
   - Pins a uniform `"num_ctx": 4096` so the KV cache stays small in VRAM and
     Ollama does not reload the model between calls.
3. **No-CPU-Spill Guard (`/api/ps`):**
   On an 8 GB GPU (e.g., RTX A4000 Laptop), PyTorch embeddings + Cross-Encoder
   use ~0.8 GB VRAM. A 6.1 GB model (`gemma4:e4b-it-qat`) fits in the remaining
   VRAM, whereas a 7.6 GB model (`gemma4:12b`) spills to system RAM/CPU and
   becomes painfully slow. Checking `/api/ps` (`size_vram` vs `size`) lets us
   detect CPU spill and fall back to cloud automatically.
4. **Lazy OpenAI client initialization:**
   `OpenAI()` is only constructed if cloud fallback is actually needed, so
   local-only runs work even without `OPENAI_API_KEY` set.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from openai import OpenAI

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:e4b-it-qat"
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_OLLAMA_NUM_CTX = 4096


@dataclass(frozen=True)
class LLMResult:
    """Response payload paired with which backend actually answered."""

    content: str
    provider: str  # "ollama" or "openai"
    model: str
    fallback_reason: str | None = None


class LocalFirstLLMClient:
    """Tries local Ollama first (enforcing 100% GPU residency) and falls back to OpenAI."""

    def __init__(
        self,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
        openai_model: str | None = None,
        ollama_num_ctx: int = DEFAULT_OLLAMA_NUM_CTX,
        require_gpu_only: bool = True,
        timeout_seconds: int = 120,
    ) -> None:
        raw_url = (
            ollama_base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
        )
        self.ollama_base_url = raw_url.removesuffix("/v1").rstrip("/")
        self.ollama_model = (
            ollama_model or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        )
        self.openai_model = (
            openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        )
        self.ollama_num_ctx = ollama_num_ctx
        self.require_gpu_only = require_gpu_only
        self.timeout_seconds = timeout_seconds
        self._openai_client: OpenAI | None = None

    def _get_openai_client(self) -> OpenAI:
        """Lazily instantiate the OpenAI client only when cloud fallback is triggered."""
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    def _post_json(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.ollama_base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_json(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.ollama_base_url}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def check_ollama_vram_fit(self, model_name: str) -> tuple[bool, str]:
        """Inspect Ollama's `/api/ps` to verify the loaded model is 100% in GPU VRAM.

        Returns:
            `(fits_in_vram, human_detail)` — returns `(True, ...)` if `size_vram >= 99%`
            of `size` or if `/api/ps` cannot report sizes.
        """
        try:
            ps_data = self._get_json("/api/ps")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return True, "ps unavailable"

        for entry in ps_data.get("models", []):
            name = entry.get("name", "") or entry.get("model", "")
            if name == model_name or name.split(":")[0] == model_name.split(":")[0]:
                size = int(entry.get("size", 0) or 0)
                size_vram = int(entry.get("size_vram", 0) or 0)
                if size > 0:
                    ratio = size_vram / size
                    vram_gb = size_vram / (1024**3)
                    total_gb = size / (1024**3)
                    detail = f"{vram_gb:.2f}/{total_gb:.2f} GB in VRAM ({ratio:.0%})"
                    return ratio >= 0.99, detail
        return True, "model not listed in /api/ps"

    def _call_ollama(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.1,
        num_predict: int = 1024,
    ) -> str:
        payload: dict = {
            "model": self.ollama_model,
            "stream": False,
            "think": False,
            "messages": messages,
            "options": {
                "num_ctx": self.ollama_num_ctx,
                "num_predict": num_predict,
                "temperature": temperature,
            },
        }
        if json_mode:
            payload["format"] = "json"

        data = self._post_json("/api/chat", payload)
        content = data.get("message", {}).get("content", "")
        if not content or not content.strip():
            raise RuntimeError("Ollama returned an empty response")

        if json_mode:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise RuntimeError("Ollama JSON response was not an object")

        if self.require_gpu_only:
            fits_vram, detail = self.check_ollama_vram_fit(self.ollama_model)
            if not fits_vram:
                raise RuntimeError(
                    f"CPU spill detected for {self.ollama_model}: {detail}"
                )

        return content

    def _call_openai(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
    ) -> str:
        kwargs: dict = {
            "model": self.openai_model,
            "messages": messages,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._get_openai_client().chat.completions.create(**kwargs)
        return response.choices[0].message.content or ("{}" if json_mode else "")

    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        prefer_local: bool = True,
        num_predict: int = 1024,
    ) -> LLMResult:
        """Send chat messages using local Ollama first, falling back to OpenAI on error or CPU spill."""
        fallback_reason: str | None = None

        if prefer_local:
            try:
                content = self._call_ollama(
                    messages, json_mode=json_mode, num_predict=num_predict
                )
                return LLMResult(
                    content=content,
                    provider="ollama",
                    model=self.ollama_model,
                )
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                RuntimeError,
                OSError,
            ) as exc:
                fallback_reason = str(exc)
                print(
                    f"⚠ Local Ollama ({self.ollama_model}) unavailable or spilled to CPU "
                    f"({fallback_reason}); falling back to OpenAI ({self.openai_model})."
                )

        content = self._call_openai(messages, json_mode=json_mode)
        return LLMResult(
            content=content,
            provider="openai",
            model=self.openai_model,
            fallback_reason=fallback_reason,
        )

    def expand_query(self, query: str, max_variants: int = 2) -> list[str]:
        """Generate alternative search queries using synonyms and related technical concepts.

        Why Multi-Query Expansion helps RAG recall:
        -------------------------------------------
        Users often phrase questions differently from textbook chapter titles
        (e.g., asking "how to remove duplicates in a list" when the book documents
        deduplication under "Sets" and "unique elements"). Generating 2 short
        alternative queries bridges that vocabulary gap before FAISS + BM25 search.
        """
        prompt = (
            f"Rewrite the search query below into {max_variants} short alternative search "
            "queries that use synonyms or closely related technical concepts to improve "
            "document retrieval.\n\n"
            f"Original query: {query}\n\n"
            'Reply with a single JSON object only: {"queries": ["query 1", "query 2"]}'
        )
        try:
            result = self.chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
                num_predict=150,
            )
            parsed = json.loads(result.content or "{}")
            raw_queries = parsed.get("queries", [])
            if not isinstance(raw_queries, list):
                return []

            variants: list[str] = []
            seen = {query.strip().lower()}
            for item in raw_queries:
                text = str(item).strip()
                if text and text.lower() not in seen:
                    seen.add(text.lower())
                    variants.append(text)
                    if len(variants) >= max_variants:
                        break
            return variants
        except Exception:  # noqa: BLE001 - expansion is best-effort; never block retrieval
            return []
