"""Unit tests for Prompt Generator."""

import unittest
from typing import Type
from pydantic import BaseModel

from llm.base import BaseLLMProvider, LLMResponse
from prompts.generator import PromptGenerator, PromptVariant, GeneratedVariantsBundle


class MockGeneratorProvider(BaseLLMProvider):
    """Deterministic mock provider for testing variant generator."""

    def __init__(self, bundle: GeneratedVariantsBundle):
        self.bundle = bundle

    def generate_text(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(content="", model_name="mock-model", latency_ms=10.0)

    def generate_structured(self, prompt: str, response_schema: Type[BaseModel], **kwargs):
        return self.bundle, LLMResponse(content="{}", model_name="mock-model", latency_ms=20.0)

    def is_available(self) -> bool:
        return True


class TestPromptGenerator(unittest.TestCase):
    def test_generate_variants_success(self):
        bundle = GeneratedVariantsBundle(
            variants=[
                PromptVariant(
                    strategy_name="Role-Based",
                    strategy_description="Assigns senior analyst persona",
                    prompt_text="You are a senior analyst. Summarize...",
                    expected_benefit="Domain focus and authority",
                    potential_limitations="May be too formal",
                ),
                PromptVariant(
                    strategy_name="Constraint-Based",
                    strategy_description="Strict negative guardrails",
                    prompt_text="Summarize in under 100 words. Do NOT include opinions.",
                    expected_benefit="Conciseness and strict adherence",
                    potential_limitations="Might truncate subtle nuance",
                ),
            ]
        )
        provider = MockGeneratorProvider(bundle)
        generator = PromptGenerator(provider=provider)

        variants, resp = generator.generate_variants(
            original_prompt="Summarize this text",
            task_type="Summarization",
            num_variants=2,
        )

        self.assertEqual(len(variants), 2)
        self.assertEqual(variants[0].strategy_name, "Role-Based")
        self.assertEqual(variants[1].strategy_name, "Constraint-Based")
        self.assertEqual(resp.model_name, "mock-model")

    def test_empty_prompt_rejected(self):
        dummy_variant = PromptVariant(
            strategy_name="Dummy",
            strategy_description="Desc",
            prompt_text="Text",
            expected_benefit="Benefit",
            potential_limitations="Limit",
        )
        provider = MockGeneratorProvider(GeneratedVariantsBundle(variants=[dummy_variant]))
        generator = PromptGenerator(provider=provider)
        with self.assertRaises(ValueError):
            generator.generate_variants("")

    def test_generated_prompts_do_not_contain_placeholders(self):
        """Verify generated prompts do not contain source-data placeholders or slot variables."""
        from llm.mock_provider import MockLLMProvider
        from prompts.generator import FORBIDDEN_PLACEHOLDER_SUBSTRINGS

        provider = MockLLMProvider()
        generator = PromptGenerator(provider=provider)

        variants, _ = generator.generate_variants(
            original_prompt="Summarize this text",
            task_type="Summarization",
            num_variants=3,
        )

        for variant in variants:
            p_text_lower = variant.prompt_text.lower()
            for forbidden in FORBIDDEN_PLACEHOLDER_SUBSTRINGS:
                self.assertNotIn(
                    forbidden.lower(),
                    p_text_lower,
                    f"Variant '{variant.strategy_name}' contained forbidden placeholder '{forbidden}' in text: {variant.prompt_text}",
                )

    def test_generator_sanitizes_llm_placeholders(self):
        """Verify generator automatically sanitizes placeholders if returned by an LLM."""
        from prompts.generator import FORBIDDEN_PLACEHOLDER_SUBSTRINGS, sanitize_prompt_text

        raw_test_prompts = [
            "You are an expert executive news analyst...\nArticle text: [INSERT ARTICLE HERE]\nSummarize concisely.",
            "Summarize this:\n[PASTE TEXT HERE]\nDo not extrapolate.",
            "Summarize {article} for an executive.",
            "Extract metrics from {source_data}.",
            "Please INSERT ARTICLE and summarize it.",
            "Please INSERT TEXT and review.",
            "Document text: [INSERT DOCUMENT]",
        ]

        bundle = GeneratedVariantsBundle(
            variants=[
                PromptVariant(
                    strategy_name=f"Strategy-{i}",
                    strategy_description="Test",
                    prompt_text=raw,
                    expected_benefit="Benefit",
                    potential_limitations="None",
                )
                for i, raw in enumerate(raw_test_prompts)
            ]
        )

        provider = MockGeneratorProvider(bundle)
        generator = PromptGenerator(provider=provider)

        cleaned_variants, _ = generator.generate_variants(
            original_prompt="Test prompt",
            num_variants=len(raw_test_prompts),
        )

        for var in cleaned_variants:
            lower = var.prompt_text.lower()
            for forbidden in [
                "[INSERT",
                "[PASTE",
                "INSERT ARTICLE",
                "INSERT TEXT",
                "{source_data}",
                "{article}",
            ]:
                self.assertNotIn(
                    forbidden.lower(),
                    lower,
                    f"Forbidden token '{forbidden}' found in sanitized text: {var.prompt_text}",
                )


if __name__ == "__main__":
    unittest.main()
