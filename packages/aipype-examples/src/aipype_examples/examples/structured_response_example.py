"""Example demonstrating LLMTask structured response support with Pydantic models.

This example shows how to:
1. Define Pydantic models for structured data extraction
2. Use llm() helper with response_format parameter
3. Extract and validate structured data from unstructured text
4. Build a pipeline agent with @task decorators
"""

import json
import os
import sys
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from aipype import PipelineAgent, task, llm, LLMTask
from aipype import print_header, print_message_box


# Define Pydantic models for structured data
class CompanyInfo(BaseModel):
    """Structured information about a company."""

    name: str = Field(description="Company name")
    industry: str = Field(description="Primary industry sector")
    founded_year: int = Field(description="Year the company was founded")
    headquarters: str = Field(description="Headquarters location")
    key_products: List[str] = Field(description="Main products or services")


class DataExtractionAgent(PipelineAgent):
    """Agent that extracts structured data from unstructured text using @task syntax."""

    @task
    def extract_company(self) -> LLMTask:
        """Extract company information from text using structured output.

        Demonstrates using response_format parameter with Pydantic model.
        The LLM will return data matching the CompanyInfo schema.
        """
        provider = self.config.get("llm_provider", "openai")
        model = self.config.get("llm_model", "gpt-4o-mini")
        company_text = self.config.get(
            "company_text",
            """
            Tesla Inc. is an American electric vehicle and clean energy company
            founded in 2003. Headquartered in Austin, Texas, Tesla designs and
            manufactures electric cars, battery energy storage, solar panels,
            and related products. Their flagship products include the Model S,
            Model 3, Model X, and Model Y vehicles, as well as the Powerwall
            home battery system.
            """,
        )

        return llm(
            prompt=f"Extract company information from this text: {company_text}",
            model=model,
            provider=provider,
            response_format=CompanyInfo,  # Pydantic model for structured output
            temperature=0.1,  # Low temperature for factual extraction
        )


def display_structured_data(agent: DataExtractionAgent) -> None:
    """Display extracted structured data in a formatted way."""
    if not agent.context:
        print("No data to display")
        return

    # Get structured data from task result
    company_result: Dict[str, Any] = agent.context.get_result("extract_company") or {}

    # Display company info
    if company_result and "parsed_object" in company_result:
        company = company_result["parsed_object"]
        print_message_box(
            "COMPANY INFORMATION",
            [
                f"Name: {company.get('name')}",
                f"Industry: {company.get('industry')}",
                f"Founded: {company.get('founded_year')}",
                f"Headquarters: {company.get('headquarters')}",
                f"Products: {', '.join(company.get('key_products', []))}",
            ],
        )


def main() -> None:
    """Run the structured response extraction example."""
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY environment variable is not set!")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr add it to your .env file.")
        sys.exit(1)

    print_header("STRUCTURED RESPONSE EXTRACTION EXAMPLE")

    print_message_box(
        "STRUCTURED OUTPUTS WITH PYDANTIC",
        [
            "This example demonstrates:",
            "- Using Pydantic models to define output schemas",
            "- Using llm() helper with response_format parameter",
            "- Extracting structured data from unstructured text",
            "- Accessing parsed objects via result['parsed_object']",
            "",
            "Supported providers: OpenAI (gpt-4o), Anthropic (Claude 3+)",
        ],
    )

    # Create and run agent
    agent = DataExtractionAgent(
        "data_extractor",
        {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "enable_parallel": False,  # Sequential for clarity
            "stop_on_failure": True,
        },
    )

    agent.run()

    print()
    display_structured_data(agent)

    # Print raw extracted data for verification
    print()
    print_header("RAW EXTRACTED DATA (for verification)")

    # Get parsed objects directly using the correct path
    company_parsed = agent.context.get_path_value("extract_company.parsed_object")

    if company_parsed:
        print("\nCompany Info:")
        print(json.dumps(company_parsed, indent=2))
    else:
        print("\nCompany Info: [FAILED OR NO DATA]")

    print()
    print_message_box(
        "STRUCTURED EXTRACTION COMPLETE",
        [
            "Successfully extracted:",
            "- Company information (name, industry, products)",
        ],
    )


if __name__ == "__main__":
    main()
