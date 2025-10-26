"""Integration tests for LLMTask structured responses with real LLM providers.

These tests make actual API calls to LLM providers and require valid API keys.
Tests are skipped if required environment variables are not set.
"""

import pytest
import os
from pydantic import BaseModel

from aipype import LLMTask


class NewsArticle(BaseModel):
    """Pydantic model for news article extraction."""

    headline: str
    summary: str
    key_topics: list[str]


class PersonInfo(BaseModel):
    """Pydantic model for person information."""

    name: str
    age: int
    occupation: str
    hobbies: list[str]


class RecipeInfo(BaseModel):
    """Pydantic model for recipe extraction."""

    name: str
    cuisine: str
    ingredients: list[str]
    prep_time_minutes: int


class Moon(BaseModel):
    """Pydantic model for moon information."""

    name: str
    diameter_km: int  # Approximate diameter in kilometers


class Planet(BaseModel):
    """Pydantic model for planet with its moons."""

    name: str
    position: int  # Position from the sun (1-8)
    planet_type: str  # "terrestrial" or "gas giant" or "ice giant"
    major_moons: list[Moon]  # List of major moons


class SolarSystemInfo(BaseModel):
    """Pydantic model for solar system planets and moons."""

    total_planets: int
    planets: list[Planet]


# Check for API keys
OPENAI_AVAILABLE = os.getenv("OPENAI_API_KEY") is not None
ANTHROPIC_AVAILABLE = os.getenv("ANTHROPIC_API_KEY") is not None


@pytest.mark.integration
class TestOpenAIStructuredResponses:
    """Integration tests for OpenAI structured responses."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    def test_openai_pydantic_article_extraction(self) -> None:
        """Test OpenAI with Pydantic model for article extraction."""
        task = LLMTask(
            "extract_article",
            {
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "prompt": """Extract structured information from this news text:

                Breaking News: Scientists Discover New Species in Amazon Rainforest

                A team of researchers has discovered a previously unknown species of frog
                in the Amazon rainforest. The discovery highlights the importance of
                biodiversity conservation in threatened ecosystems. The research was
                published in Nature magazine this week.
                """,
                "response_format": NewsArticle,
                "temperature": 0.3,
            },
        )

        result = task.run()

        assert result.is_success() is True
        assert "parsed_object" in result.data
        assert "headline" in result.data["parsed_object"]
        assert "summary" in result.data["parsed_object"]
        assert "key_topics" in result.data["parsed_object"]
        assert isinstance(result.data["parsed_object"]["key_topics"], list)
        assert len(result.data["parsed_object"]["key_topics"]) > 0

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    def test_openai_pydantic_person_extraction(self) -> None:
        """Test OpenAI with Pydantic model for person info extraction."""
        task = LLMTask(
            "extract_person",
            {
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "prompt": """Extract person information from this bio:

                Dr. Sarah Chen is a 42-year-old marine biologist working at the Oceanographic
                Research Institute. In her free time, she enjoys scuba diving, photography,
                and playing the violin. She has published over 50 research papers on coral
                reef ecosystems.
                """,
                "response_format": PersonInfo,
                "temperature": 0.1,
            },
        )

        result = task.run()

        assert result.is_success() is True
        assert "parsed_object" in result.data
        parsed = result.data["parsed_object"]

        assert "name" in parsed
        assert "Sarah Chen" in parsed["name"]
        assert parsed["age"] == 42
        assert "occupation" in parsed
        assert "biologist" in parsed["occupation"].lower()
        assert isinstance(parsed["hobbies"], list)
        assert len(parsed["hobbies"]) >= 2

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    def test_openai_json_schema_format(self) -> None:
        """Test OpenAI with manual JSON schema format."""
        task = LLMTask(
            "extract_recipe",
            {
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "prompt": """Extract recipe information:

                Classic Italian Margherita Pizza

                This traditional Italian pizza takes about 30 minutes to prepare.
                You'll need: pizza dough, fresh mozzarella, tomato sauce, fresh basil,
                olive oil, and salt.
                """,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Recipe",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "cuisine": {"type": "string"},
                                "ingredients": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "prep_time_minutes": {"type": "integer"},
                            },
                            "required": [
                                "name",
                                "cuisine",
                                "ingredients",
                                "prep_time_minutes",
                            ],
                        },
                        "strict": True,
                    },
                },
                "temperature": 0.2,
            },
        )

        result = task.run()

        assert result.is_success() is True
        assert "parsed_object" in result.data
        parsed = result.data["parsed_object"]

        assert "name" in parsed
        assert "pizza" in parsed["name"].lower()
        assert "cuisine" in parsed
        assert "italian" in parsed["cuisine"].lower()
        assert isinstance(parsed["ingredients"], list)
        assert len(parsed["ingredients"]) >= 4
        assert parsed["prep_time_minutes"] > 0

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    def test_openai_question_with_nested_structures(self) -> None:
        """Test OpenAI answering a question directly with nested structured output."""
        task = LLMTask(
            "planets_and_moons",
            {
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "prompt": """List all 8 planets in our solar system in order from the sun,
                along with their major moons (moons with diameter > 1000 km).
                Include the approximate diameter for each moon.""",
                "response_format": SolarSystemInfo,
                "temperature": 0.1,
            },
        )

        result = task.run()

        assert result.is_success() is True
        assert "parsed_object" in result.data
        parsed = result.data["parsed_object"]

        # Verify basic structure
        assert parsed["total_planets"] == 8
        assert isinstance(parsed["planets"], list)
        assert len(parsed["planets"]) == 8

        # Verify first planet (Mercury - no major moons)
        mercury = parsed["planets"][0]
        assert mercury["name"] == "Mercury"
        assert mercury["position"] == 1
        assert len(mercury["major_moons"]) == 0

        # Verify Jupiter has major moons (should have at least 4 Galilean moons)
        jupiter = next((p for p in parsed["planets"] if p["name"] == "Jupiter"), None)
        assert jupiter is not None
        assert len(jupiter["major_moons"]) >= 4

        # Verify moon structure
        if jupiter["major_moons"]:
            first_moon = jupiter["major_moons"][0]
            assert "name" in first_moon
            assert "diameter_km" in first_moon
            assert isinstance(first_moon["diameter_km"], int)
            assert first_moon["diameter_km"] > 1000


@pytest.mark.integration
class TestAnthropicStructuredResponses:
    """Integration tests for Anthropic structured responses."""

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="ANTHROPIC_API_KEY not set")
    def test_anthropic_pydantic_extraction(self) -> None:
        """Test Anthropic/Claude with Pydantic model."""
        task = LLMTask(
            "extract_article",
            {
                "llm_provider": "anthropic",
                "llm_model": "claude-3-5-haiku-20241022",
                "prompt": """Extract article information from this text:

                Tech Giants Announce AI Safety Partnership

                Leading technology companies have formed a new alliance focused on AI safety
                and responsible development. The partnership aims to establish common standards
                and share research on AI alignment and risk mitigation.
                """,
                "response_format": NewsArticle,
                "temperature": 0.3,
                "max_tokens": 500,
            },
        )

        result = task.run()

        assert result.is_success() is True
        assert "parsed_object" in result.data
        parsed = result.data["parsed_object"]

        assert "headline" in parsed
        assert (
            "ai" in parsed["headline"].lower() or "safety" in parsed["headline"].lower()
        )
        assert "summary" in parsed
        assert len(parsed["summary"]) > 20
        assert isinstance(parsed["key_topics"], list)
        assert len(parsed["key_topics"]) > 0


@pytest.mark.integration
class TestStructuredResponseErrors:
    """Test error handling for structured responses."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    def test_invalid_response_format_type(self) -> None:
        """Test task with invalid response_format type."""
        task = LLMTask(
            "invalid_format",
            {
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "prompt": "Extract data",
                "response_format": "invalid_string",  # Invalid type
            },
        )

        result = task.run()

        # Should fail validation
        assert result.is_error() is True

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    def test_model_usage_tracking_with_structured_output(self) -> None:
        """Test that usage tracking works with structured outputs."""
        task = LLMTask(
            "usage_test",
            {
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "prompt": "Extract: John Doe is 30 years old and works as a teacher. Hobbies: reading, hiking.",
                "response_format": PersonInfo,
                "temperature": 0.1,
            },
        )

        result = task.run()

        assert result.is_success() is True
        assert "usage" in result.data
        assert "prompt_tokens" in result.data["usage"]
        assert "completion_tokens" in result.data["usage"]
        assert "total_tokens" in result.data["usage"]
        assert result.data["usage"]["total_tokens"] > 0
