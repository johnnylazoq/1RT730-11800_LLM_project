"""LLM client wrapper for different providers."""

import json
import os

from pydantic import BaseModel


class LLMClient:
    """Wrapper for LLM calls (OpenAI, Anthropic, etc.)."""

    def __init__(self, model: str, temperature: float = 0.7, api_key: str | None = None):
        """Initialize LLM client.

        Args:
            model: Model name (e.g., gpt-4-turbo, claude-3-5-sonnet-20241022).
            temperature: Sampling temperature.
            api_key: API key (optional, loads from env if not provided).
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.provider = self._detect_provider()
        self.client = None
        self._init_client()

    def _detect_provider(self) -> str:
        """Detect LLM provider from model name."""
        if "gpt" in self.model.lower():
            return "openai"
        elif "claude" in self.model.lower():
            return "anthropic"
        else:
            return "unknown"

    def _init_client(self):
        """Initialize client for detected provider."""
        if self.provider == "openai":
            try:
                from openai import OpenAI

                api_key = self.api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment")
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai not installed. Install with: pip install openai")

        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic

                api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in environment")
                self.client = Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic not installed. Install with: pip install anthropic")
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def generate(self, prompt: str, structured_output: BaseModel | None = None) -> str | dict:
        """Generate text or structured output.

        Args:
            prompt: Input prompt.
            structured_output: Pydantic model for structured output (optional).

        Returns:
            Generated text or parsed object.
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, structured_output)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, structured_output)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_openai(self, prompt: str, structured_output: BaseModel | None = None):
        """Generate using OpenAI API."""
        if structured_output:
            # Use JSON mode for structured output
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=structured_output,
            )
            return response.choices[0].message.parsed
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content

    def _generate_anthropic(self, prompt: str, structured_output: BaseModel | None = None):
        """Generate using Anthropic API."""
        # Anthropic doesn't have native structured output, use JSON parsing
        system_prompt = "You are a helpful assistant. Respond with valid JSON."

        if structured_output:
            system_prompt += f"\n\nRespond with JSON matching this schema:\n{structured_output.model_json_schema()}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        text = response.content[0].text

        # Parse JSON if structured output expected
        if structured_output:
            try:
                # Extract JSON from response
                import re

                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    data = json.loads(json_str)
                    return structured_output(**data)
            except (json.JSONDecodeError, ValueError):
                pass

        return text
