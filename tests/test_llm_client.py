"""Unit tests for LocalFirstLLMClient (Ollama primary + no-CPU-spill guard + OpenAI fallback)."""

from __future__ import annotations

import json
import unittest
import urllib.error
from unittest.mock import patch

from pdf_rag.llm_client import LocalFirstLLMClient


class TestLocalFirstLLMClient(unittest.TestCase):
    def test_uses_local_ollama_when_100_percent_in_vram(self) -> None:
        client = LocalFirstLLMClient(ollama_model="gemma4:e4b-it-qat")
        expected_json = json.dumps(
            {"answer": "Relief valve blockage.", "sources": ["[1]"], "confidence": 0.92}
        )

        with (
            patch.object(
                client,
                "_post_json",
                return_value={"message": {"content": expected_json}},
            ),
            patch.object(
                client,
                "_get_json",
                return_value={
                    "models": [
                        {
                            "name": "gemma4:e4b-it-qat",
                            "size": 6_100_000_000,
                            "size_vram": 6_100_000_000,
                        }
                    ]
                },
            ),
            patch.object(client, "_call_openai") as mock_openai,
        ):
            result = client.chat([{"role": "user", "content": "hi"}], json_mode=True)

        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.model, "gemma4:e4b-it-qat")
        self.assertIsNone(result.fallback_reason)
        mock_openai.assert_not_called()

    def test_falls_back_to_openai_when_ollama_spills_to_cpu(self) -> None:
        client = LocalFirstLLMClient(
            ollama_model="gemma4:12b",
            openai_model="gpt-6-luna",
            require_gpu_only=True,
        )
        valid_json = json.dumps({"answer": "ok", "sources": [], "confidence": 0.5})

        with (
            patch.object(
                client,
                "_post_json",
                return_value={"message": {"content": valid_json}},
            ),
            patch.object(
                client,
                "_get_json",
                return_value={
                    "models": [
                        {
                            "name": "gemma4:12b",
                            "size": 8_000_000_000,
                            "size_vram": 5_600_000_000,
                        }
                    ]
                },
            ),
            patch.object(
                client,
                "_call_openai",
                return_value=valid_json,
            ) as mock_openai,
        ):
            result = client.chat([{"role": "user", "content": "hi"}], json_mode=True)

        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.model, "gpt-6-luna")
        self.assertIn("CPU spill detected", result.fallback_reason or "")
        mock_openai.assert_called_once()

    def test_falls_back_to_openai_when_ollama_unreachable(self) -> None:
        client = LocalFirstLLMClient(openai_model="gpt-6-luna")
        valid_json = json.dumps(
            {"answer": "cloud", "sources": ["[1]"], "confidence": 0.9}
        )

        with (
            patch.object(
                client,
                "_post_json",
                side_effect=urllib.error.URLError("connection refused"),
            ),
            patch.object(
                client,
                "_call_openai",
                return_value=valid_json,
            ) as mock_openai,
        ):
            result = client.chat([{"role": "user", "content": "hi"}], json_mode=True)

        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.model, "gpt-6-luna")
        self.assertIsNotNone(result.fallback_reason)
        mock_openai.assert_called_once()

    def test_expand_query_returns_deduplicated_variants(self) -> None:
        client = LocalFirstLLMClient()
        payload = json.dumps(
            {
                "queries": [
                    "How to remove duplicates in a list in Python?",
                    "Python remove duplicate items using set",
                    "Python unique elements from sequence",
                    "extra ignored variant",
                ]
            }
        )
        with (
            patch.object(
                client,
                "_post_json",
                return_value={"message": {"content": payload}},
            ),
            patch.object(
                client,
                "check_ollama_vram_fit",
                return_value=(True, "100%"),
            ),
        ):
            variants = client.expand_query(
                "How to remove duplicates in a list in Python?", max_variants=2
            )

        self.assertEqual(
            variants,
            [
                "Python remove duplicate items using set",
                "Python unique elements from sequence",
            ],
        )


if __name__ == "__main__":
    unittest.main()
