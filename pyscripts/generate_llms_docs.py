#!/usr/bin/env python3
"""
Intelligent LLMs.txt Documentation Generator

This script uses aipype to generate concise, example-focused documentation
in Markdown format suitable for LLM training and understanding.

Output structure:
1. Core Concepts (1-2 paragraphs)
2. Task Examples (syntactically correct Python code)
3. API Reference (1-liner descriptions)

Target size: <25KB per package

Requirements:
    Set the appropriate API key for the LLM provider:
    - For Claude Haiku 4.5 (default): export ANTHROPIC_API_KEY='your-key'
    - For OpenAI: export OPENAI_API_KEY='your-key'

Usage:
    # Generate for default packages (aipype, aipype-g)
    python pyscripts/generate_llms_docs.py

    # Generate for specific package
    python pyscripts/generate_llms_docs.py --packages aipype

    # Use different LLM model
    python pyscripts/generate_llms_docs.py --llm-model gpt-4o-mini --llm-provider openai

    # Generate multiple packages
    python pyscripts/generate_llms_docs.py --packages aipype,aipype-extras,aipype-g

Output:
    Files are created in docs/_build/ directory:
    - docs/_build/llms-aipype-ai.txt
    - docs/_build/llms-aipype-g-ai.txt
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, override

from aipype import (
    BaseTask,
    ConditionalTask,
    DependencyType,
    LLMTask,
    PipelineAgent,
    TaskDependency,
    TaskResult,
    TransformTask,
    print_header,
)


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_BUILD_DIR = PROJECT_ROOT / "docs" / "_build"
DOCS_TEXT_DIR = DOCS_BUILD_DIR / "text"
EXAMPLES_DIR = PROJECT_ROOT / "packages" / "aipype-examples" / "src" / "aipype_examples"

# Size limit (25KB)
MAX_SIZE_BYTES = 25600


class BuildRawDocsTask(BaseTask):
    """Execute the build_docs_txt.sh script to generate raw Sphinx documentation."""

    @override
    def run(self) -> TaskResult:
        """Run the documentation build script."""
        try:
            script_path = PROJECT_ROOT / "scripts" / "build_docs_txt.sh"

            # Run the script
            result = subprocess.run(
                [str(script_path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                return TaskResult.failure(
                    f"BuildRawDocsTask failed: Script exited with code {result.returncode}",
                    metadata={"stderr": result.stderr, "stdout": result.stdout},
                )

            return TaskResult.success(
                data={"stdout": result.stdout, "stderr": result.stderr}
            )

        except subprocess.TimeoutExpired:
            return TaskResult.failure("BuildRawDocsTask failed: Script timeout after 120s")
        except Exception as e:
            return TaskResult.failure(
                f"BuildRawDocsTask failed: {str(e)}",
                metadata={"error_type": type(e).__name__},
            )


class ReadFilesTask(BaseTask):
    """Read multiple files from disk."""

    @override
    def run(self) -> TaskResult:
        """Read files specified in config."""
        try:
            file_paths = self.config.get("file_paths", [])
            files_data = {}

            for file_path in file_paths:
                path = Path(file_path)
                if not path.exists():
                    return TaskResult.failure(
                        f"ReadFilesTask failed: File not found: {file_path}"
                    )

                try:
                    content = path.read_text(encoding="utf-8")
                    files_data[str(file_path)] = content
                except Exception as e:
                    return TaskResult.failure(
                        f"ReadFilesTask failed: Error reading {file_path}: {str(e)}"
                    )

            return TaskResult.success(data={"files": files_data})

        except Exception as e:
            return TaskResult.failure(
                f"ReadFilesTask failed: {str(e)}",
                metadata={"error_type": type(e).__name__},
            )


class SaveFileTask(BaseTask):
    """Save content to a file."""

    @override
    def run(self) -> TaskResult:
        """Save content from context to file."""
        try:
            file_path = self.config.get("file_path")
            content = self.config.get("content", "")

            if not file_path:
                return TaskResult.failure("SaveFileTask failed: file_path not specified")

            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

            file_size = path.stat().st_size
            return TaskResult.success(
                data={
                    "file_path": str(file_path),
                    "size_bytes": file_size,
                    "size_kb": round(file_size / 1024, 2),
                }
            )

        except Exception as e:
            return TaskResult.failure(
                f"SaveFileTask failed: {str(e)}",
                metadata={"error_type": type(e).__name__},
            )


class DocsGeneratorAgent(PipelineAgent):
    """Agent that generates concise, example-focused LLM documentation."""

    @override
    def setup_tasks(self) -> List[BaseTask]:
        """Set up the documentation generation pipeline."""
        package_name = self.config.get("package_name", "aipype")
        llm_provider = self.config.get("llm_provider", "openai")
        llm_model = self.config.get(
            "llm_model", "anthropic/claude-haiku-4-5-20251001"
        )

        # Convert package name: aipype-g -> aipype_g for file lookup
        package_file = package_name.replace("-", "_")
        raw_doc_path = DOCS_TEXT_DIR / "api" / f"{package_file}.txt"
        output_path = DOCS_BUILD_DIR / f"llms-{package_name}-ai.txt"

        # Example files to include
        example_files = self._get_example_files()

        return [
            # Step 1: Build raw Sphinx docs
            BuildRawDocsTask("build_raw_docs", {}),
            # Step 2: Read raw documentation
            ReadFilesTask(
                "read_raw_docs",
                {"file_paths": [str(raw_doc_path)]},
                [
                    TaskDependency(
                        "build_result", "build_raw_docs.stdout", DependencyType.REQUIRED
                    )
                ],
            ),
            # Step 2b: Extract raw docs content
            self._create_extract_docs_content_task(str(raw_doc_path)),
            # Step 3: Read example files
            ReadFilesTask("read_examples", {"file_paths": example_files}),
            # Step 3b: Extract examples content
            self._create_extract_examples_content_task(example_files),
            # Step 4: Extract core concepts (1-2 paragraphs)
            self._create_extract_core_concepts_task(
                llm_provider, llm_model, package_name
            ),
            # Step 5: Extract task examples (syntactically correct)
            self._create_extract_task_examples_task(llm_provider, llm_model, package_name),
            # Step 6: Extract API reference (1-liners)
            self._create_extract_api_reference_task(
                llm_provider, llm_model, package_name
            ),
            # Step 7: Combine into final Markdown
            self._create_combine_markdown_task(package_name),
            # Step 8: Validate size (<25KB)
            self._create_validate_size_task(),
            # Step 9: Save to file
            SaveFileTask(
                "save_docs",
                {"file_path": str(output_path)},
                [
                    TaskDependency(
                        "content",
                        "combine_markdown.final_markdown",
                        DependencyType.REQUIRED,
                    ),
                    TaskDependency(
                        "validation",
                        "validate_size.data",
                        DependencyType.OPTIONAL,
                    ),
                ],
            ),
        ]

    def _get_example_files(self) -> List[str]:
        """Get list of example files to include."""
        example_files = [
            EXAMPLES_DIR / "tutorial" / "01_basic_print_agent.py",
            EXAMPLES_DIR / "tutorial" / "02_llm_task.py",
            EXAMPLES_DIR / "tutorial" / "03_dependent_tasks.py",
            EXAMPLES_DIR / "tutorial" / "04_conditional_task.py",
            EXAMPLES_DIR / "examples" / "structured_response_example.py",
        ]
        return [str(f) for f in example_files if f.exists()]

    def _create_extract_docs_content_task(self, raw_doc_path: str) -> TransformTask:
        """Extract raw docs content from files dict."""

        def extract_content(files: Dict[str, str]) -> str:
            """Extract the content of the raw doc file."""
            return files.get(raw_doc_path, "")

        return TransformTask(
            "extract_docs_content",
            {
                "transform_function": extract_content,
                "input_field": "files",
                "output_name": "raw_docs",
            },
            [TaskDependency("files", "read_raw_docs.files", DependencyType.REQUIRED)],
        )

    def _create_extract_examples_content_task(
        self, example_files: List[str]
    ) -> TransformTask:
        """Extract examples content from files dict."""

        def extract_and_combine(files: Dict[str, str]) -> str:
            """Combine all example files into one string."""
            combined = []
            for file_path in example_files:
                content = files.get(file_path, "")
                if content:
                    combined.append(f"=== {Path(file_path).name} ===\n{content}\n")
            return "\n".join(combined)

        return TransformTask(
            "extract_examples_content",
            {
                "transform_function": extract_and_combine,
                "input_field": "files",
                "output_name": "examples_content",
            },
            [TaskDependency("files", "read_examples.files", DependencyType.REQUIRED)],
        )

    def _create_extract_core_concepts_task(
        self, llm_provider: str, llm_model: str, package_name: str
    ) -> LLMTask:
        """Create task to extract core concepts."""
        prompt = """Based on the raw documentation provided, write a comprehensive 2-3 paragraph introduction explaining the core concepts of the {package_name} framework.

Focus on:
- PipelineAgent and task orchestration
- TaskContext and data sharing
- TaskDependency for wiring tasks
- TaskResult pattern for error handling
- Parallel execution and task phases

Raw documentation:
${{raw_docs}}

Output a clear, comprehensive explanation suitable for developers learning the framework. Use Markdown formatting."""

        return LLMTask(
            "extract_core_concepts",
            {
                "prompt_template": prompt.format(package_name=package_name),
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "temperature": 0.2,
                "max_tokens": 1500,
            },
            [
                TaskDependency(
                    "raw_docs", "extract_docs_content.raw_docs", DependencyType.REQUIRED
                )
            ],
        )

    def _create_extract_task_examples_task(
        self, llm_provider: str, llm_model: str, package_name: str
    ) -> LLMTask:
        """Create task to extract and create concise examples."""
        prompt = """Based on the example files provided, create comprehensive, syntactically correct Python code examples for the main task types in {package_name}.

For each task type (LLMTask, SearchTask, TransformTask, ConditionalTask, etc.), create a minimal but COMPLETE and RUNNABLE example showing:
- Necessary imports
- Task creation
- Key configuration parameters
- Usage pattern

IMPORTANT: Also include a comprehensive pipeline example showing:
- Multiple tasks with dependencies (TaskDependency)
- How data flows between tasks using TaskContext
- Both REQUIRED and OPTIONAL dependency types
- A complete PipelineAgent implementation

Example files:
${{examples_content}}

Requirements:
1. Code must be syntactically correct and runnable
2. Use proper Python code blocks with ```python
3. Keep individual task examples minimal but complete
4. Include the pipeline example to show task orchestration
5. Include brief comments explaining key parts
6. Format as Markdown with headers for each task type (## Task Examples, ### LLMTask, ### Pipeline Example, etc.)

Output the examples section in Markdown format."""

        return LLMTask(
            "extract_task_examples",
            {
                "prompt_template": prompt.format(package_name=package_name),
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "temperature": 0.2,
                "max_tokens": 6000,
            },
            [
                TaskDependency(
                    "examples_content", "extract_examples_content.examples_content", DependencyType.REQUIRED
                )
            ],
        )

    def _create_extract_api_reference_task(
        self, llm_provider: str, llm_model: str, package_name: str
    ) -> LLMTask:
        """Create task to extract API reference."""
        prompt = """Based on the raw documentation, create a comprehensive API reference for all important classes and their methods in {package_name}.

Include ALL important classes such as:
- Agent classes (PipelineAgent)
- Task classes (BaseTask, LLMTask, SearchTask, TransformTask, ConditionalTask, etc.)
- Result classes (TaskResult, AgentRunResult)
- Dependency classes (TaskDependency, DependencyResolver)
- Context classes (TaskContext)
- Planning classes (TaskExecutionPlan)

For EACH class, provide:
1. Brief class description (1-2 sentences)
2. All important methods with their signatures
3. Important attributes/properties
4. Key configuration parameters if applicable

Format as hierarchical Markdown structure:

### ClassName

Brief class description explaining its purpose and role.

**Methods:**
- `method_name(param1: type, param2: type) -> return_type` - Brief description of what the method does
- `another_method() -> type` - Description

**Attributes:**
- `attribute_name` - Description (if applicable)

**Configuration Parameters:**
- `param_name` (type) - Description (if applicable)

Raw documentation:
${{raw_docs}}

IMPORTANT: Be comprehensive - include all major classes and their key methods. Use proper method signatures with parameter types. Output the complete API reference section in Markdown format with header (## API Reference)."""

        return LLMTask(
            "extract_api_reference",
            {
                "prompt_template": prompt.format(package_name=package_name),
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "temperature": 0.2,
                "max_tokens": 8000,
            },
            [
                TaskDependency(
                    "raw_docs", "extract_docs_content.raw_docs", DependencyType.REQUIRED
                )
            ],
        )

    def _create_combine_markdown_task(self, package_name: str) -> TransformTask:
        """Create task to combine sections into final Markdown."""

        def combine_sections(
            core_concepts: str, examples: str, api_reference: str
        ) -> str:
            """Combine all sections into final Markdown document."""
            markdown = f"""# {package_name} Documentation

{core_concepts}

{examples}

{api_reference}

---
*Generated with aipype DocsGeneratorAgent*
"""
            return markdown.strip()

        return TransformTask(
            "combine_markdown",
            {
                "transform_function": combine_sections,
                "input_fields": ["core_concepts", "examples", "api_reference"],
                "output_name": "final_markdown",
            },
            [
                TaskDependency(
                    "core_concepts",
                    "extract_core_concepts.content",
                    DependencyType.REQUIRED,
                ),
                TaskDependency(
                    "examples", "extract_task_examples.content", DependencyType.REQUIRED
                ),
                TaskDependency(
                    "api_reference",
                    "extract_api_reference.content",
                    DependencyType.REQUIRED,
                ),
            ],
        )

    def _create_validate_size_task(self) -> ConditionalTask:
        """Create task to validate output size."""

        def check_size(final_markdown: str) -> bool:
            """Check if markdown is under size limit."""
            size_bytes = len(final_markdown.encode("utf-8"))
            return size_bytes <= MAX_SIZE_BYTES

        def handle_size_exceeded(final_markdown: str) -> Dict[str, Any]:
            """Handle case where size limit is exceeded."""
            size_bytes = len(final_markdown.encode("utf-8"))
            size_kb = round(size_bytes / 1024, 2)
            raise RuntimeError(
                f"Documentation size {size_kb}KB exceeds limit of {MAX_SIZE_BYTES / 1024}KB. "
                "Please adjust LLM prompts to generate more concise output."
            )

        return ConditionalTask(
            "validate_size",
            {
                "condition_function": check_size,
                "condition_inputs": ["final_markdown"],
                "then_function": lambda markdown: {  # pyright: ignore
                    "validation_result": "passed",
                    "size_bytes": len(markdown.encode("utf-8")),  # pyright: ignore
                    "size_kb": round(len(markdown.encode("utf-8")) / 1024, 2),  # pyright: ignore
                },
                "else_function": handle_size_exceeded,
                "skip_reason": "Size validation skipped",
            },
            [
                TaskDependency(
                    "final_markdown",
                    "combine_markdown.final_markdown",
                    DependencyType.REQUIRED,
                )
            ],
        )


def main() -> int:
    """Main entry point for the documentation generator."""
    parser = argparse.ArgumentParser(
        description="Generate concise LLM-friendly documentation using aipype"
    )
    parser.add_argument(
        "--packages",
        type=str,
        default="aipype,aipype-g",
        help="Comma-separated list of packages to process (default: aipype,aipype-g)",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="anthropic/claude-haiku-4-5-20251001",
        help="LLM model to use (default: anthropic/claude-haiku-4-5-20251001)",
    )

    args = parser.parse_args()

    # Parse packages
    packages = [pkg.strip() for pkg in args.packages.split(",")]

    print_header("INTELLIGENT LLMS.TXT DOCUMENTATION GENERATOR")
    print(f"\nPackages: {', '.join(packages)}")
    print(f"LLM: {args.llm_model}")
    print(f"Output: docs/_build/llms-{{package}}-ai.txt\n")

    # Process each package
    for package in packages:
        print(f"\n{'='*60}")
        print(f"Processing: {package}")
        print(f"{'='*60}\n")

        agent = DocsGeneratorAgent(
            f"docs-generator-{package}",
            {
                "package_name": package,
                "llm_provider": args.llm_provider,
                "llm_model": args.llm_model,
            },
        )

        agent.run()

        # Check if successful
        if agent.context:
            save_result = agent.context.get_path_value("save_docs.data")
            if save_result:
                print(f"\nGenerated: {save_result.get('file_path')}")
                print(f"   Size: {save_result.get('size_kb')}KB")
            else:
                print(f"\nFailed to generate documentation for {package}")
                agent.display_results()
                return 1

    print(f"\n{'='*60}")
    print("All packages processed successfully!")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
