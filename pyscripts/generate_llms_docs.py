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
    Files are created in docs/_build/ directory (3 files per package):
    - docs/_build/llms-aipype-{version}-concepts-ai.txt
    - docs/_build/llms-aipype-{version}-examples-ai.txt
    - docs/_build/llms-aipype-{version}-api-ai.txt
    - docs/_build/llms-aipype-g-{version}-concepts-ai.txt
    - docs/_build/llms-aipype-g-{version}-examples-ai.txt
    - docs/_build/llms-aipype-g-{version}-api-ai.txt

    Version is read from each package's pyproject.toml file.
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


def get_package_version(package_name: str) -> str:
    """Read version from package's pyproject.toml file.

    Args:
        package_name: Name of the package (e.g., "aipype", "aipype-g")

    Returns:
        Version string (e.g., "0.1.0a4")
    """
    pyproject_path = PROJECT_ROOT / "packages" / package_name / "pyproject.toml"

    if not pyproject_path.exists():
        return "unknown"

    try:
        content = pyproject_path.read_text(encoding="utf-8")
        # Find version line: version = "0.1.0a4"
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("version"):
                # Extract version string between quotes
                version = line.split("=")[1].strip().strip('"').strip("'")
                return version
    except Exception:
        return "unknown"

    return "unknown"


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
            return TaskResult.failure(
                "BuildRawDocsTask failed: Script timeout after 120s"
            )
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
                return TaskResult.failure(
                    "SaveFileTask failed: file_path not specified"
                )

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
        llm_model = self.config.get("llm_model", "anthropic/claude-haiku-4-5-20251001")

        # Get package version
        version = get_package_version(package_name)

        # Convert package name: aipype-g -> aipype_g for file lookup
        package_file = package_name.replace("-", "_")
        raw_doc_path = DOCS_TEXT_DIR / "api" / f"{package_file}.txt"

        # Output paths for 3 separate files with version
        concepts_path = (
            DOCS_BUILD_DIR / f"llms-{package_name}-{version}-concepts-ai.txt"
        )
        examples_path = (
            DOCS_BUILD_DIR / f"llms-{package_name}-{version}-examples-ai.txt"
        )
        api_path = DOCS_BUILD_DIR / f"llms-{package_name}-{version}-api-ai.txt"

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
            self._create_extract_task_examples_task(
                llm_provider, llm_model, package_name
            ),
            # Step 6: Extract API reference (1-liners)
            self._create_extract_api_reference_task(
                llm_provider, llm_model, package_name
            ),
            # Step 7: Validate concepts size
            self._create_validate_task(
                "validate_concepts", "content", "extract_core_concepts.content"
            ),
            # Step 8: Save concepts file
            SaveFileTask(
                "save_concepts",
                {"file_path": str(concepts_path)},
                [
                    TaskDependency(
                        "content",
                        "extract_core_concepts.content",
                        DependencyType.REQUIRED,
                    ),
                    TaskDependency(
                        "validation",
                        "validate_concepts.data",
                        DependencyType.OPTIONAL,
                    ),
                ],
            ),
            # Step 9: Validate examples size
            self._create_validate_task(
                "validate_examples", "content", "extract_task_examples.content"
            ),
            # Step 10: Save examples file
            SaveFileTask(
                "save_examples",
                {"file_path": str(examples_path)},
                [
                    TaskDependency(
                        "content",
                        "extract_task_examples.content",
                        DependencyType.REQUIRED,
                    ),
                    TaskDependency(
                        "validation",
                        "validate_examples.data",
                        DependencyType.OPTIONAL,
                    ),
                ],
            ),
            # Step 11: Validate API reference size
            self._create_validate_task(
                "validate_api", "content", "extract_api_reference.content"
            ),
            # Step 12: Save API reference file
            SaveFileTask(
                "save_api",
                {"file_path": str(api_path)},
                [
                    TaskDependency(
                        "content",
                        "extract_api_reference.content",
                        DependencyType.REQUIRED,
                    ),
                    TaskDependency(
                        "validation",
                        "validate_api.data",
                        DependencyType.OPTIONAL,
                    ),
                ],
            ),
        ]

    def _get_example_files(self) -> List[str]:
        """Get list of example files to include based on package."""
        package_name = self.config.get("package_name", "aipype")

        # Define package-specific example files
        if package_name == "aipype":
            # Core framework examples
            example_files = [
                EXAMPLES_DIR / "tutorial" / "01_basic_print_agent.py",
                EXAMPLES_DIR / "tutorial" / "02_llm_task.py",
                EXAMPLES_DIR / "tutorial" / "03_dependent_tasks.py",
                EXAMPLES_DIR / "tutorial" / "04_conditional_task.py",
                EXAMPLES_DIR / "examples" / "structured_response_example.py",
            ]
        elif package_name == "aipype-g":
            # Google API integration examples
            example_files = [
                EXAMPLES_DIR / "examples" / "gmail_test_agent.py",
            ]
        elif package_name == "aipype-extras":
            # No examples for aipype-extras yet - return empty list
            # The LLM will work from API docs only
            example_files = []
        else:
            # Default: try to find any examples
            example_files = []

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
        prompt = """Based on the raw documentation and example files provided, create comprehensive, syntactically correct Python code examples for the main components in {package_name}.

Create minimal but COMPLETE and RUNNABLE examples showing:
- Necessary imports
- Component creation and configuration
- Key configuration parameters
- Usage patterns

IMPORTANT: If example files are provided, use them as reference but create cleaner, more concise versions. If no example files are provided, create examples based on the raw documentation.

For packages with tasks (like aipype), include a comprehensive pipeline example showing:
- Multiple tasks with dependencies (TaskDependency)
- How data flows between tasks using TaskContext
- Both REQUIRED and OPTIONAL dependency types
- A complete PipelineAgent implementation

Raw documentation:
${{raw_docs}}

Example files (may be empty):
${{examples_content}}

Requirements:
1. Code must be syntactically correct and runnable
2. Use proper Python code blocks with ```python
3. Keep examples minimal but complete
4. Include brief comments explaining key parts
5. Format as Markdown with headers (## Usage Examples, ### Component Name, etc.)
6. If no examples are provided, create them from the API documentation

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
                    "raw_docs", "extract_docs_content.raw_docs", DependencyType.REQUIRED
                ),
                TaskDependency(
                    "examples_content",
                    "extract_examples_content.examples_content",
                    DependencyType.OPTIONAL,
                ),
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

    def _create_validate_task(
        self, task_name: str, content_name: str, source_path: str
    ) -> ConditionalTask:
        """Create task to validate output size.

        Args:
            task_name: Name for the validation task (e.g., "validate_concepts")
            content_name: Name of the content input parameter (e.g., "content")
            source_path: Path to content in context (e.g., "extract_core_concepts.content")
        """

        def check_size(content: str) -> bool:
            """Check if content is under size limit."""
            size_bytes = len(content.encode("utf-8"))
            return size_bytes <= MAX_SIZE_BYTES

        def handle_size_exceeded(content: str) -> Dict[str, Any]:
            """Handle case where size limit is exceeded."""
            size_bytes = len(content.encode("utf-8"))
            size_kb = round(size_bytes / 1024, 2)
            raise RuntimeError(
                f"Documentation size {size_kb}KB exceeds limit of {MAX_SIZE_BYTES / 1024}KB. "
                "Please adjust LLM prompts to generate more concise output."
            )

        return ConditionalTask(
            task_name,
            {
                "condition_function": check_size,
                "condition_inputs": [content_name],
                "then_function": lambda content: {  # pyright: ignore
                    "validation_result": "passed",
                    "size_bytes": len(content.encode("utf-8")),  # pyright: ignore
                    "size_kb": round(len(content.encode("utf-8")) / 1024, 2),  # pyright: ignore
                },
                "else_function": lambda content: handle_size_exceeded(
                    content
                ),  # Fixed: wrapped in lambda
                "skip_reason": "Size validation skipped",
            },
            [
                TaskDependency(
                    content_name,
                    source_path,
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
    print(
        "Output: docs/_build/llms-{package}-{version}-[concepts|examples|api]-ai.txt\n"
    )

    # Process each package
    for package in packages:
        print(f"\n{'=' * 60}")
        print(f"Processing: {package}")
        print(f"{'=' * 60}\n")

        agent = DocsGeneratorAgent(
            f"docs-generator-{package}",
            {
                "package_name": package,
                "llm_provider": args.llm_provider,
                "llm_model": args.llm_model,
            },
        )

        agent.run()

        # Get package version for file path construction
        version = get_package_version(package)

        # Check if successful by verifying all 3 files were generated
        concepts_path = DOCS_BUILD_DIR / f"llms-{package}-{version}-concepts-ai.txt"
        examples_path = DOCS_BUILD_DIR / f"llms-{package}-{version}-examples-ai.txt"
        api_path = DOCS_BUILD_DIR / f"llms-{package}-{version}-api-ai.txt"

        if concepts_path.exists() and examples_path.exists() and api_path.exists():
            # Get file sizes
            concepts_size = round(concepts_path.stat().st_size / 1024, 2)
            examples_size = round(examples_path.stat().st_size / 1024, 2)
            api_size = round(api_path.stat().st_size / 1024, 2)

            print(f"\n✅ Generated 3 files for {package}:")
            print(f"   📝 Concepts: {concepts_path} ({concepts_size}KB)")
            print(f"   📚 Examples: {examples_path} ({examples_size}KB)")
            print(f"   📖 API Reference: {api_path} ({api_size}KB)")
        else:
            print(f"\n❌ Failed to generate documentation for {package}")
            agent.display_results()
            return 1

    print(f"\n{'=' * 60}")
    print("All packages processed successfully!")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
