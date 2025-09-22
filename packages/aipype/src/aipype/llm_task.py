"""Advanced LLM task with context-aware prompts, tool calling, and template substitution.

This module provides LLMTask, a sophisticated task for integrating Large Language Models
into AI pipelines. It supports automatic prompt template substitution using dependency
data, tool/function calling, multiple LLM providers, and comprehensive response handling.

Key Features:
    * **Template Substitution**: Dynamic prompt generation using ${variable} syntax
    * **Tool Calling**: Function calling with automatic tool registration and execution
    * **Multi-Provider**: Support for OpenAI, Anthropic, Ollama, and other LLM providers
    * **Context Integration**: Automatic injection of dependency data into prompts
    * **Response Handling**: Structured response parsing and error handling
    * **Logging**: Optional detailed logging for debugging and monitoring

**Template System**

LLMTask uses ${variable} syntax for dynamic prompt generation:

.. code-block:: python

    task = LLMTask("summarize", {
        "prompt": "Summarize these articles: ${articles}",
        "context": "You are an expert ${domain} analyst",
        "llm_provider": "openai",
        "llm_model": "gpt-4"
    }, dependencies=[
        TaskDependency("articles", "search.results", REQUIRED),
        TaskDependency("domain", "config.analysis_domain", OPTIONAL, default_value="general")
    ])

**Tool Calling**

Enable function calling with automatic tool management:

.. code-block:: python

    @tool
    def search_web(query: str) -> Dict[str, Any]:
        return {"results": web_search(query)}

    task = LLMTask("research", {
        "prompt": "Research ${topic} and provide analysis",
        "tools": [search_web],
        "llm_provider": "openai",
        "llm_model": "gpt-4"
    })

**Supported Providers**

* **OpenAI**: gpt-4, gpt-3.5-turbo, etc.
* **Anthropic**: claude-3-opus, claude-3-sonnet, etc.
* **Ollama**: Local models (llama2, mistral, etc.)
* **Google**: gemini-pro, gemini-pro-vision
* **Azure OpenAI**: Azure-hosted OpenAI models
* **And many more via litellm**

**Configuration Options**

**Core settings:**

* llm_provider: Provider name (required)
* llm_model: Model identifier (required)
* prompt/prompt_template: Main prompt text with templates
* temperature: Randomness control (0.0-2.0, default: 0.7)
* max_tokens: Maximum response tokens (default: 1000)
* timeout: Request timeout in seconds (default: 60)

**Tool calling:**

* tools: List of @tool decorated functions
* tool_choice: "auto", "none", or specific tool name
* parallel_tool_calls: Enable parallel tool execution
* max_tool_execution_time: Tool timeout in seconds

**Context and role:**

* context: System/context message (supports templates)
* role: User role description (supports templates)

**Response Format**

LLMTask returns structured data including:

* **content**: Generated text response
* **usage**: Token usage statistics
* **tool_calls**: List of executed tool calls (if any)
* **model**: Model used for generation
* **provider**: Provider used
* **metadata**: Additional response information

**Example Usage**

**Basic text generation:**

.. code-block:: python

    task = LLMTask("generate", {
        "prompt": "Write a brief summary about ${topic}",
        "llm_provider": "openai",
        "llm_model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 500
    })

**With dependencies and templates:**

.. code-block:: python

    task = LLMTask("analyze", {
        "prompt": "Analyze these search results about ${topic}: ${results}",
        "context": "You are an expert ${field} researcher",
        "llm_provider": "anthropic",
        "llm_model": "claude-3-opus"
    }, dependencies=[
        TaskDependency("topic", "input.topic", REQUIRED),
        TaskDependency("results", "search.results", REQUIRED),
        TaskDependency("field", "config.research_field", OPTIONAL, default_value="general")
    ])

**With tool calling:**

.. code-block:: python

    @tool
    def calculate(expression: str) -> float:
        return eval(expression)  # Simple calculator

    task = LLMTask("math_helper", {
        "prompt": "Help solve this math problem: ${problem}",
        "tools": [calculate],
        "llm_provider": "openai",
        "llm_model": "gpt-4",
        "tool_choice": "auto"
    })

**Error Handling**

LLMTask handles various error conditions gracefully:

* API failures (network, authentication, rate limits)
* Invalid model/provider combinations
* Tool execution errors
* Template substitution errors
* Response parsing failures

All errors are returned as TaskResult.failure() with detailed error messages
and relevant metadata for debugging.

See Also:
    * BaseTask: Parent class for task implementation patterns
    * TaskDependency: For injecting data into prompt templates
    * @tool decorator: For creating tool functions
    * litellm: Underlying LLM provider abstraction
"""

import re
import os
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypedDict

from typing import override
from dotenv import load_dotenv
import litellm
from .base_task import BaseTask
from .task_result import TaskResult
from .task_dependencies import TaskDependency
from .task_context import TaskContext
from .tool_registry import ToolRegistry
from .tool_executor import ToolExecutor

# Load environment variables from .env file if present
load_dotenv()


class ChatMessage(TypedDict):
    """Type definition for chat messages."""

    role: str
    content: str
    tool_calls: Optional[List[Any]]


class LLMTask(BaseTask):
    """Advanced LLM integration task with template substitution and tool calling.

    LLMTask provides sophisticated Large Language Model integration for AI pipelines.
    It supports dynamic prompt generation through template substitution, automatic
    function/tool calling, and seamless integration with multiple LLM providers.

    The task automatically resolves template variables from dependency data and
    manages the complete LLM interaction lifecycle including request formatting,
    response parsing, tool execution, and error handling.

    Core Capabilities:
        * **Dynamic Prompts**: ${variable} template substitution from dependencies
        * **Tool Integration**: Automatic function calling with @tool decorated functions
        * **Multi-Provider**: Support for 50+ LLM providers via litellm
        * **Response Processing**: Structured response parsing and validation
        * **Error Recovery**: Graceful handling of API failures and timeouts
        * **Usage Tracking**: Detailed token usage and cost monitoring

    Template Substitution:
        Templates use ${variable_name} syntax and are resolved from:
        1. Task dependencies (via TaskDependency)
        2. Task configuration values
        3. Environment variables (as fallback)

        Template fields support:
        * prompt/prompt_template: Main user message
        * context: System/context message
        * role: User role description

    Tool Calling Workflow:
        1. Functions decorated with @tool are automatically registered
        2. LLM decides which tools to call based on prompt
        3. Tools execute with validated parameters
        4. Results injected back into conversation
        5. LLM continues with tool results

**Provider Configuration**

Different providers require different configuration:

**OpenAI**::

    {
        "llm_provider": "openai",
        "llm_model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000
    }

**Anthropic**::

    {
        "llm_provider": "anthropic",
        "llm_model": "claude-3-opus-20240229",
        "temperature": 0.5,
        "max_tokens": 2000
    }

**Ollama (local)**::

    {
        "llm_provider": "ollama",
        "llm_model": "llama2",
        "temperature": 0.8
    }

**Azure OpenAI**::

    {
        "llm_provider": "azure",
        "llm_model": "deployment-name",
        "api_base": "https://your-resource.openai.azure.com/",
        "api_version": "2023-05-15"
    }

**Response Structure**

Successful responses include:

.. code-block:: python

    {
        "content": "Generated text response",
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 300,
            "total_tokens": 450
        },
        "tool_calls": [
            {
                "tool_name": "search_web",
                "arguments": {"query": "AI trends"},
                "result": {"results": [...]},
                "success": True
            }
        ],
        "model": "gpt-4",
        "provider": "openai",
        "metadata": {
            "response_time": 2.34,
            "template_variables": ["topic", "domain"]
        }
    }

**Common Usage Patterns**

**Content generation with templates:**

.. code-block:: python

    LLMTask("writer", {
        "prompt": "Write a ${length} article about ${topic} for ${audience}",
        "context": "You are an expert ${field} writer",
        "llm_provider": "openai",
        "llm_model": "gpt-4",
        "temperature": 0.7
    }, dependencies=[
        TaskDependency("topic", "input.topic", REQUIRED),
        TaskDependency("length", "config.article_length", OPTIONAL, default_value="brief"),
        TaskDependency("audience", "config.target_audience", REQUIRED),
        TaskDependency("field", "config.expertise_field", REQUIRED)
    ])

**Research with tool calling:**

.. code-block:: python

    @tool
    def search_academic_papers(query: str, limit: int = 5) -> List[Dict]:
        # Search implementation
        return papers

    LLMTask("researcher", {
        "prompt": "Research ${topic} and provide comprehensive analysis",
        "tools": [search_academic_papers],
        "llm_provider": "anthropic",
        "llm_model": "claude-3-opus",
        "tool_choice": "auto",
        "max_tokens": 3000
    })

**Data analysis and reasoning:**

.. code-block:: python

    LLMTask("analyzer", {
        "prompt": "Analyze this data and provide insights: ${data}",
        "context": "You are a data scientist specializing in ${domain}",
        "llm_provider": "openai",
        "llm_model": "gpt-4",
        "temperature": 0.3,  # Lower temperature for analytical tasks
        "max_tokens": 2000
    }, dependencies=[
        TaskDependency("data", "processor.results", REQUIRED),
        TaskDependency("domain", "config.analysis_domain", REQUIRED)
    ])

**Error Handling**

Common failure scenarios and recovery:

* **API Errors**: Network failures, authentication issues
* **Rate Limits**: Automatic retry with exponential backoff
* **Invalid Models**: Clear error messages for unsupported models
* **Tool Failures**: Individual tool errors don't fail entire task
* **Template Errors**: Missing variables result in clear error messages

**Performance Considerations**

* Use appropriate max_tokens to control costs and latency
* Lower temperature (0.1-0.3) for factual/analytical tasks
* Higher temperature (0.7-1.0) for creative tasks
* Consider model capabilities vs cost (gpt-3.5-turbo vs gpt-4)
* Use tool_choice="none" to disable tool calling when not needed

See Also:
    * @tool: For creating callable functions
    * TaskDependency: For template variable injection
    * litellm: Underlying provider abstraction
    * Tool calling examples in the examples package
    """

    REQUIRED_CONFIGS = ["llm_provider", "llm_model"]
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize an advanced LLM task with template and tool support.

        Creates a new LLM task that can automatically substitute template variables
        from dependencies, execute function calls, and interact with multiple LLM
        providers. The task handles the complete LLM interaction lifecycle.

        Args:
            name: Unique identifier for this task within the pipeline. Used for
                logging, dependency references, and result storage.

            config: LLM configuration dictionary with the following keys:

                **Required:**

                - llm_provider (str): LLM provider name. Supported values include:
                  "openai", "anthropic", "ollama", "azure", "google", "cohere", etc.
                - llm_model (str): Model identifier specific to the provider:
                  - OpenAI: "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"
                  - Anthropic: "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"
                  - Ollama: "llama2", "mistral", "codellama"
                  - Google: "gemini-pro", "gemini-pro-vision"

                **Prompt Configuration:**

                - prompt OR prompt_template (str): Main user message with ${var} templates
                - context (str, optional): System/context message (supports templates)
                - role (str, optional): User role description (supports templates)

                **Generation Parameters:**

                - temperature (float, 0.0-2.0): Randomness control (default: 0.7)
                - max_tokens (int): Maximum response tokens (default: 1000)
                - timeout (float): Request timeout in seconds (default: 60)

                **Tool Calling:**

                - tools (List[Callable], optional): List of @tool decorated functions
                - tool_choice (str, optional): "auto", "none", or specific tool name
                - parallel_tool_calls (bool): Enable parallel tool execution (default: True)
                - max_tool_execution_time (float): Tool timeout in seconds (default: 30)

                **Provider-Specific:**

                - api_key (str, optional): API key (can also use environment variables)
                - api_base (str, optional): Custom API endpoint (for Azure, local deployments)
                - api_version (str, optional): API version (for Azure OpenAI)

            dependencies: List of TaskDependency objects for template variable injection.
                Template variables in prompt, context, and role fields will be
                automatically replaced with resolved dependency values.

        Example:
            Basic text generation::

                LLMTask("summarizer", {
                    "prompt": "Summarize this text: ${content}",
                    "llm_provider": "openai",
                    "llm_model": "gpt-4",
                    "temperature": 0.3,
                    "max_tokens": 500
                }, dependencies=[
                    TaskDependency("content", "input.text", REQUIRED)
                ])

            Advanced with tools and context::

                @tool
                def web_search(query: str) -> Dict[str, Any]:
                    return {"results": search_engine.search(query)}

                LLMTask("researcher", {
                    "prompt": "Research ${topic} and provide analysis",
                    "context": "You are an expert ${field} researcher with access to web search",
                    "llm_provider": "anthropic",
                    "llm_model": "claude-3-opus",
                    "tools": [web_search],
                    "tool_choice": "auto",
                    "temperature": 0.5,
                    "max_tokens": 3000
                }, dependencies=[
                    TaskDependency("topic", "config.research_topic", REQUIRED),
                    TaskDependency("field", "config.expertise_field", REQUIRED)
                ])

            Local model with Ollama::

                LLMTask("local_chat", {
                    "prompt": "Help with this question: ${question}",
                    "llm_provider": "ollama",
                    "llm_model": "llama2",
                    "temperature": 0.8,
                    "max_tokens": 2000
                })

        Template Variables:
            Variables in prompt, context, and role fields use ${variable_name} syntax.
            Resolution order:
            1. Dependency values (from TaskDependency objects)
            2. Direct config values
            3. Environment variables
            4. Error if required variable not found

        Tool Integration:
            Functions decorated with @tool are automatically:
            1. Registered with the LLM provider
            2. Made available to the model during generation
            3. Executed when called by the model
            4. Results injected back into the conversation

        Error Handling:
            Configuration errors are caught during validation:
            - Missing required fields (llm_provider, llm_model)
            - Invalid parameter ranges (temperature, max_tokens)
            - Unsupported tool configurations
            - Provider-specific validation

        Environment Variables:
            API keys can be provided via environment variables:
            - OPENAI_API_KEY for OpenAI
            - ANTHROPIC_API_KEY for Anthropic
            - GOOGLE_API_KEY for Google
            - Etc.

        See Also:
            - @tool: For creating callable functions
            - TaskDependency: For template variable injection
            - Provider documentation for model-specific capabilities
        """
        super().__init__(name, config, dependencies)
        self.validation_rules = {
            "required": ["llm_provider", "llm_model"],
            "optional": [
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "max_tool_execution_time",
            ],
            "defaults": {
                "temperature": self.DEFAULT_TEMPERATURE,
                "max_tokens": self.DEFAULT_MAX_TOKENS,
                "timeout": self.DEFAULT_TIMEOUT,
                "context": "",
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "max_tool_execution_time": 30.0,
            },
            "types": {
                "llm_provider": str,
                "llm_model": str,
                "temperature": (int, float),
                "max_tokens": int,
                "timeout": (int, float),
                "context": str,
                # tools type validation handled by custom validator
                "tool_choice": str,
                "parallel_tool_calls": bool,
                "max_tool_execution_time": (int, float),
            },
            "ranges": {
                "temperature": (0, 2),
                "max_tokens": (1, 32000),
                "timeout": (1, 600),
                "max_tool_execution_time": (1, 300),
            },
            # Validation lambdas intentionally use dynamic typing for flexibility across field types
            "custom": {
                "llm_provider": lambda x: x.strip() != "",  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
                "llm_model": lambda x: x.strip() != "",  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
                "tools": self._validate_tools,
                "tool_choice": lambda x: x in ["auto", "none"] or isinstance(x, str),  # pyright: ignore[reportUnknownLambdaType]
            },
        }
        self.context_instance: Optional[TaskContext] = None
        self.resolved_prompt: Optional[str] = None
        self.resolved_context: Optional[str] = None
        self.resolved_role: Optional[str] = None
        self.logs_file = os.getenv("MI_AGENT_LLM_LOGS_FILE")

        # Tool-related instance variables
        self.tool_registry: Optional[ToolRegistry] = None
        self.tool_executor: Optional[ToolExecutor] = None
        self.supports_tools: bool = False
        self.tool_setup_error: Optional[str] = None

        # Apply tool-related defaults when tools are provided
        if "tools" in self.config and self.config["tools"]:
            # Apply defaults for tool-related configuration
            tool_defaults = {
                "tool_choice": "auto",
                "max_tool_execution_time": 30.0,
            }

            # Only add parallel_tool_calls for providers that support it (not Ollama)
            provider = self.config.get("llm_provider", "openai").lower()
            if provider != "ollama":
                tool_defaults["parallel_tool_calls"] = True

            for key, default_value in tool_defaults.items():
                if key not in self.config:
                    self.config[key] = default_value

            # Setup tools - store errors for later handling in run()
            try:
                self._setup_tool_support()
            except Exception as e:
                # Store the error for later - don't raise exception during init
                self.tool_setup_error = f"LLMTask tool setup operation failed: {str(e)}"
                # Leave supports_tools as False so run() can handle the error

        # Set up litellm configuration
        if self.config.get("timeout"):
            litellm.request_timeout = self.config["timeout"]  # pyright: ignore

        # Configure litellm logging to use our logger
        litellm.set_verbose = False  # pyright: ignore

    def _get_api_key(self) -> Optional[str]:
        """Get API key from config or environment variables."""
        # First try explicit config
        if "api_key" in self.config:
            return (
                str(self.config["api_key"])
                if self.config["api_key"] is not None
                else None
            )

        # Then try environment variables based on provider
        provider = self.config["llm_provider"].lower()

        env_key_map: Dict[str, Optional[str]] = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "cohere": "COHERE_API_KEY",
            "palm": "PALM_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "azure": "AZURE_API_KEY",
            "ollama": None,  # Ollama typically doesn't require API key
            "huggingface": "HUGGINGFACE_API_KEY",
        }

        env_key: Optional[str] = env_key_map.get(provider)
        if env_key:
            return os.getenv(env_key)

        return None

    def _get_api_base(self) -> Optional[str]:
        """Get API base URL from config or environment variables."""
        # First try explicit config
        if "api_base" in self.config:
            return (
                str(self.config["api_base"])
                if self.config["api_base"] is not None
                else None
            )

        # Then try environment variables based on provider
        provider = self.config["llm_provider"].lower()

        env_base_map: Dict[str, str] = {
            "ollama": "OLLAMA_API_BASE",
            "openai": "OPENAI_API_BASE",
            "anthropic": "ANTHROPIC_API_BASE",
            "claude": "ANTHROPIC_API_BASE",
        }

        env_key = env_base_map.get(provider)
        if env_key:
            return os.getenv(env_key)

        return None

    def _validate_tools(self, tools: List[Callable[..., Any]]) -> bool:
        """Validate tools configuration.

        Args:
            tools: List of tool functions to validate

        Returns:
            True if all tools are valid
        """
        # Runtime validation needed for dynamic config data
        if not isinstance(tools, list):  # pyright: ignore[reportUnnecessaryIsInstance]
            return False

        for tool in tools:
            if not callable(tool):
                return False
            if not hasattr(tool, "_is_tool"):
                return False

        return True

    def _setup_tool_support(self) -> None:
        """Initialize tool registry and executor from function list."""
        tool_functions = self.config["tools"]

        # Validate all functions are decorated with @tool
        for func in tool_functions:
            if not hasattr(func, "_is_tool"):
                func_name = getattr(func, "__name__", str(func))
                raise ValueError(
                    f"Function {func_name} must be decorated with @tool. "
                    f"Add '@tool' decorator above the function definition."
                )

        # Create registry and executor
        max_exec_time = self.config.get("max_tool_execution_time", 30.0)
        self.tool_registry = ToolRegistry(tool_functions)
        self.tool_executor = ToolExecutor(self.tool_registry, max_exec_time)
        self.supports_tools = True

        self.logger.info(
            f"Initialized {len(tool_functions)} tools for task '{self.name}'"
        )

    def _prepare_messages_with_tool_context(self) -> List[Dict[str, Any]]:
        """Enhanced message preparation with automatic tool context."""
        messages: List[Dict[str, Any]] = []

        # Build system context with tool information
        context_parts: List[str] = []

        # Add original context
        original_context = self.resolved_context or self.config.get("context")
        if original_context:
            context_parts.append(original_context)

        # Add tool context automatically if tools are available
        if self.supports_tools and self.tool_registry:
            tool_context = self.tool_registry.generate_tool_context()
            if tool_context:
                context_parts.append(tool_context)

        # Add role if provided
        role = self.resolved_role or self.config.get("role")
        if role:
            context_parts.append(f"Your role: {role}")

        # Create system message with all context
        if context_parts:
            messages.append({"role": "system", "content": "\n\n".join(context_parts)})

        # Add user prompt
        prompt = self.resolved_prompt or self.config.get("prompt", "")
        messages.append({"role": "user", "content": prompt})

        return messages

    def _supports_function_calling(self) -> bool:
        """Check if current provider/model supports function calling using LiteLLM."""
        try:
            provider = self.config["llm_provider"].lower()
            model = self.config["llm_model"]

            # Use LiteLLM's built-in function calling support detection
            return litellm.utils.supports_function_calling(
                model=model,
                custom_llm_provider=provider if provider != "openai" else None,
            )
        except Exception as e:
            self.logger.warning(
                f"Could not determine function calling support: {str(e)}"
            )
            # Fallback to manual provider checking
            return self._manual_provider_check()

    def _manual_provider_check(self) -> bool:
        """Fallback manual check for function calling support."""
        provider = self.config["llm_provider"].lower()

        # Known providers that support function calling
        supported_providers = {
            "openai",
            "anthropic",
            "claude",
            "azure",
            "gemini",
            "cohere",
        }

        return provider in supported_providers

    def _handle_conversation_with_tools(
        self, initial_messages: List[Dict[str, Any]], initial_response: Any
    ) -> Any:
        """Handle multi-turn conversation with tool calls using LiteLLM."""

        messages = initial_messages.copy()
        current_response = initial_response
        max_iterations = 10  # Prevent infinite loops
        iterations = 0

        # Collect all tool call results across all iterations
        all_tool_call_results: List[Dict[str, Any]] = []

        # Handle tool calls in a loop until no more tool calls
        # LiteLLM response structure requires pyright ignore for dynamic attributes
        while (
            iterations < max_iterations
            and hasattr(current_response, "choices")
            and current_response.choices  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
            and hasattr(current_response.choices[0], "message")  # pyright: ignore[reportAttributeAccessIssue,reportUnknownArgumentType,reportUnknownMemberType]
            and hasattr(current_response.choices[0].message, "tool_calls")  # pyright: ignore[reportAttributeAccessIssue,reportUnknownArgumentType,reportUnknownMemberType]
            and current_response.choices[0].message.tool_calls  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        ):
            iterations += 1

            # Add assistant message with tool calls to conversation
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                # LiteLLM response structure requires pyright ignore for dynamic attributes
                "content": current_response.choices[0].message.content or "",  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
            }

            # Add tool_calls if they exist
            # LiteLLM response structure requires pyright ignore for dynamic attributes
            if current_response.choices[0].message.tool_calls:  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
                assistant_message["tool_calls"] = current_response.choices[  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
                    0
                ].message.tool_calls  # pyright: ignore[reportAttributeAccessIssue]

            messages.append(assistant_message)

            # Execute all tool calls and collect results
            # LiteLLM response structure requires pyright ignore for dynamic attributes
            tool_calls = current_response.choices[0].message.tool_calls  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
            for tool_call in tool_calls:  # pyright: ignore[reportUnknownVariableType]
                # Extract tool call information
                # LiteLLM tool call structure requires pyright ignore for dynamic attributes
                tool_name = tool_call.function.name  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                try:
                    # LiteLLM tool call structure requires pyright ignore for dynamic attributes
                    tool_args = json.loads(tool_call.function.arguments)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                except json.JSONDecodeError as e:
                    self.logger.error(
                        f"Invalid tool arguments JSON: {tool_call.function.arguments}"  # pyright: ignore[reportUnknownMemberType]
                    )
                    tool_result = {
                        "success": False,
                        "error": f"Invalid JSON in tool arguments: {str(e)}",
                    }
                else:
                    # Execute the tool
                    tool_result = self.tool_executor.execute_tool(tool_name, tool_args)  # type: ignore[union-attr]

                # Store tool result for final response
                all_tool_call_results.append(tool_result)

                # Add tool result message to conversation
                if tool_result.get("success"):
                    result_content = json.dumps(tool_result.get("result"))
                else:
                    result_content = tool_result.get("error", "Tool execution failed")

                tool_message: Dict[str, Any] = {
                    "role": "tool",
                    # LiteLLM tool call structure requires pyright ignore for dynamic attributes
                    "tool_call_id": tool_call.id,  # pyright: ignore[reportUnknownMemberType]
                    "content": result_content,
                }
                messages.append(tool_message)

            # Continue conversation with LiteLLM
            try:
                # Prepare LiteLLM parameters
                continuation_params: Dict[str, Any] = {
                    "model": self._build_model_name(),
                    "messages": messages,
                    "temperature": self.config.get(
                        "temperature", self.DEFAULT_TEMPERATURE
                    ),
                    "max_tokens": self.config.get(
                        "max_tokens", self.DEFAULT_MAX_TOKENS
                    ),
                }

                # Add tool parameters if tools are still available
                if self.supports_tools and self.tool_registry:
                    continuation_params["tools"] = self.tool_registry.get_tool_schemas()
                    continuation_params["tool_choice"] = self.config.get(
                        "tool_choice", "auto"
                    )
                    provider = self.config.get("llm_provider", "openai").lower()
                    if provider != "ollama" and self.config.get(
                        "parallel_tool_calls", False
                    ):
                        continuation_params["parallel_tool_calls"] = True

                # Add API credentials
                api_key = self._get_api_key()
                if api_key:
                    continuation_params["api_key"] = api_key

                api_base = self._get_api_base()
                if api_base:
                    continuation_params["api_base"] = api_base

                # Make follow-up call
                # LiteLLM completion function has dynamic signature
                current_response = litellm.completion(**continuation_params)  # pyright: ignore[reportUnknownMemberType]

            except Exception as e:
                self.logger.error(f"Error in tool conversation continuation: {str(e)}")
                # Return current response if continuation fails
                break

        if iterations >= max_iterations:
            self.logger.warning(
                f"Tool conversation reached maximum iterations ({max_iterations})"
            )

        # Store tool call results on the response object for extraction later
        current_response._tool_call_results = all_tool_call_results  # pyright: ignore[reportAttributeAccessIssue]

        return current_response

    def _prepare_messages(self) -> List[Dict[str, str]]:
        """Prepare message list for LLM API call."""
        messages: List[Dict[str, str]] = []

        # Add system context if provided
        context = self.resolved_context or self.config.get("context")
        role = self.resolved_role or self.config.get("role")

        if context or role:
            system_content: List[str] = []
            if context:
                system_content.append(context)
            if role:
                system_content.append(f"Your role: {role}")

            messages.append({"role": "system", "content": " ".join(system_content)})

        # Add the main prompt as user message
        prompt = self.resolved_prompt or self.config.get("prompt", "")
        messages.append({"role": "user", "content": prompt})

        return messages

    def _build_model_name(self) -> str:
        """Build the full model name for litellm."""
        provider = self.config["llm_provider"].lower()
        model = self.config["llm_model"]

        # For some providers, litellm expects provider/model format
        provider_prefixes = {
            "anthropic": "anthropic/",
            "claude": "anthropic/",
            "cohere": "cohere/",
            "palm": "palm/",
            "gemini": "gemini/",
            "azure": "azure/",
            "ollama": "ollama/",
            "huggingface": "huggingface/",
        }

        prefix = provider_prefixes.get(provider, "")
        return f"{prefix}{model}"

    def _make_llm_call(self) -> Any:
        """Enhanced LLM call with tool support using LiteLLM API."""
        # Use enhanced message preparation with tool context if tools are available
        if self.supports_tools:
            messages = self._prepare_messages_with_tool_context()
        else:
            messages = self._prepare_messages()

        model_name = self._build_model_name()

        # Base parameters (existing)
        call_params: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": self.config.get("temperature", self.DEFAULT_TEMPERATURE),
            "max_tokens": self.config.get("max_tokens", self.DEFAULT_MAX_TOKENS),
            "stream": self.config.get("stream", False),
        }

        # Add existing parameters
        api_key = self._get_api_key()
        if api_key:
            call_params["api_key"] = api_key

        api_base = self._get_api_base()
        if api_base:
            call_params["api_base"] = api_base
            self.logger.debug(f"Using custom API base: {api_base}")

        # Add tool support if configured and supported
        if (
            self.supports_tools
            and self.tool_registry
            and self._supports_function_calling()
        ):
            # LiteLLM directly accepts tools parameter
            call_params["tools"] = self.tool_registry.get_tool_schemas()
            call_params["tool_choice"] = self.config.get("tool_choice", "auto")

            # LiteLLM supports parallel_tool_calls parameter (but not for Ollama)
            provider = self.config.get("llm_provider", "openai").lower()
            if provider != "ollama" and self.config.get("parallel_tool_calls", False):
                call_params["parallel_tool_calls"] = True

            self.logger.debug(f"Added {len(call_params['tools'])} tools to LLM call")

        self.logger.info(f"Making LLM call to {model_name}")
        self.logger.debug(f"Messages: {len(messages)} messages")

        try:
            # LiteLLM completion function has dynamic signature
            response = litellm.completion(**call_params)  # pyright: ignore[reportUnknownMemberType]

            # Check if response contains tool calls using LiteLLM response structure
            # LiteLLM response structure requires pyright ignore for dynamic attributes
            if (
                self.supports_tools
                and hasattr(response, "choices")
                and response.choices  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
                and hasattr(response.choices[0], "message")  # pyright: ignore[reportAttributeAccessIssue,reportUnknownArgumentType,reportUnknownMemberType]
                and hasattr(response.choices[0].message, "tool_calls")  # pyright: ignore[reportAttributeAccessIssue,reportUnknownArgumentType,reportUnknownMemberType]
                and response.choices[0].message.tool_calls  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
            ):
                # Handle multi-turn conversation with tool calls
                return self._handle_conversation_with_tools(messages, response)

            self.logger.info("LLM call completed successfully")
            return response

        except Exception as e:
            error_msg = f"LLMTask API call operation failed: LLM API call to '{model_name}' failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _safe_extract_content(self, response: Any) -> str:
        """Safely extract content from LLM response, avoiding Pydantic serialization issues."""
        # Check for completely invalid responses first
        if not hasattr(response, "choices"):
            raise ValueError("Response missing 'choices' attribute")

        # LiteLLM response structure requires pyright ignore for dynamic attributes
        if not response.choices:
            raise ValueError("Response has empty choices list")

        try:
            # Try multiple ways to access content to handle different response structures
            # LiteLLM response structure requires pyright ignore for dynamic attributes
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return (
                    str(choice.message.content)
                    if choice.message.content is not None
                    else ""
                )
            elif hasattr(choice, "text"):
                return str(choice.text) if choice.text is not None else ""

            # If we can't find content in the choice, it's an error
            raise ValueError("Could not find content in response choice")

        except (IndexError, AttributeError) as e:
            # These are structural issues, not Pydantic serialization issues
            raise ValueError(f"Invalid response structure: {str(e)}") from e
        except Exception as e:
            # Log but don't re-raise for other issues (potential Pydantic issues)
            self.logger.warning(f"Non-critical error extracting content: {str(e)}")
            return ""

    def _safe_extract_usage(self, response: Any) -> Dict[str, int]:
        """Safely extract usage information from LLM response."""
        usage_info: Dict[str, int] = {}
        try:
            if hasattr(response, "usage") and response.usage is not None:
                usage = response.usage

                # Extract common usage fields safely
                if hasattr(usage, "prompt_tokens") and usage.prompt_tokens is not None:
                    usage_info["prompt_tokens"] = int(usage.prompt_tokens)
                if (
                    hasattr(usage, "completion_tokens")
                    and usage.completion_tokens is not None
                ):
                    usage_info["completion_tokens"] = int(usage.completion_tokens)
                if hasattr(usage, "total_tokens") and usage.total_tokens is not None:
                    usage_info["total_tokens"] = int(usage.total_tokens)

        except Exception as e:
            self.logger.warning(f"Error extracting usage information: {str(e)}")

        return usage_info

    def _safe_extract_model(self, response: Any) -> str:
        """Safely extract model name from LLM response."""
        try:
            if hasattr(response, "model") and response.model is not None:
                return str(response.model)

            # Fallback to config model if response doesn't have it
            return self.config.get("llm_model", "unknown")

        except Exception as e:
            self.logger.warning(f"Error extracting model: {str(e)}")
            return self.config.get("llm_model", "unknown")

    def _safe_extract_finish_reason(self, response: Any) -> Optional[str]:
        """Safely extract finish reason from LLM response."""
        try:
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if (
                    hasattr(choice, "finish_reason")
                    and choice.finish_reason is not None
                ):
                    return str(choice.finish_reason)

        except Exception as e:
            self.logger.warning(f"Error extracting finish reason: {str(e)}")

        return None

    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool call execution results from response if available.

        Args:
            response: LLM response object

        Returns:
            List of tool call results if available, empty list otherwise
        """
        # Check if response has stored tool call results
        if hasattr(response, "_tool_call_results"):
            return response._tool_call_results

        return []

    def _process_response(self, response: Any) -> Dict[str, Any]:
        """Process and extract relevant information from LLM response using safe extraction methods."""
        try:
            # Use safe extraction methods to avoid Pydantic serialization issues
            content = self._safe_extract_content(response)
            usage = self._safe_extract_usage(response)
            model = self._safe_extract_model(response)
            finish_reason = self._safe_extract_finish_reason(response)

            # Build result dictionary
            result = {
                "content": content,
                "model": model,
                "provider": self.config["llm_provider"],
            }

            # Add usage information if available
            if usage:
                result["usage"] = usage

            # Add finish reason if available
            if finish_reason is not None:
                result["finish_reason"] = finish_reason

            # Extract and include tool calls if this response used tools
            tool_calls = self._extract_tool_calls(response)
            if tool_calls:
                result["tool_calls"] = tool_calls

            self.logger.info(f"LLM response processed: {len(content)} characters")
            if "usage" in result:
                self.logger.debug(f"Token usage: {result['usage']}")

            return result

        except Exception as e:
            error_msg = f"LLMTask response processing operation failed: Failed to process LLM response: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _log_llm_interaction(
        self, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> None:
        """Log LLM input and output to file if MI_AGENT_LLM_LOGS_FILE is set.

        Args:
            input_data: Input data sent to LLM
            output_data: Output data received from LLM
        """
        if not self.logs_file:
            return

        try:
            log_entry: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "agent_name": self.agent_name,
                "task_name": self.name,
                "provider": self.config.get("llm_provider"),
                "model": self.config.get("llm_model"),
                "input": input_data,
                "output": output_data,
            }

            with open(self.logs_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        except Exception as e:
            self.logger.warning(
                f"Failed to log LLM interaction to {self.logs_file}: {str(e)}"
            )

    def _prepare_input_log_data(self) -> Dict[str, Any]:
        """Prepare input data for logging.

        Returns:
            Dictionary containing input data for logging
        """
        input_data: Dict[str, Any] = {
            "prompt": self.resolved_prompt or self.config.get("prompt", ""),
            "context": self.resolved_context or self.config.get("context"),
            "role": self.resolved_role or self.config.get("role"),
            "temperature": self.config.get("temperature", self.DEFAULT_TEMPERATURE),
            "max_tokens": self.config.get("max_tokens", self.DEFAULT_MAX_TOKENS),
            "stream": self.config.get("stream", False),
        }

        # Remove None values
        return {k: v for k, v in input_data.items() if v is not None}

    @override
    def set_context(self, context: TaskContext) -> None:
        """Set the task context for template resolution.

        Args:
            context: TaskContext instance
        """
        self.context_instance = context

    @override
    def run(self) -> TaskResult:
        """Execute the LLM task with context-resolved prompts and logging."""
        start_time = datetime.now()

        # Validate configuration using instance validation rules
        validation_failure = self._validate_or_fail(start_time)
        if validation_failure:
            return validation_failure

        # Check for tool setup errors from initialization
        if self.tool_setup_error is not None:
            self.logger.error(self.tool_setup_error)
            execution_time = (datetime.now() - start_time).total_seconds()
            return TaskResult.failure(
                error_message=self.tool_setup_error,
                metadata={"error_type": "ToolSetupError", "task_name": self.name},
                execution_time=execution_time,
            )

        try:
            self.logger.info(f"Starting contextual LLM task '{self.name}'")

            # Resolve templates before running
            self._resolve_templates()

            # Prepare input data for logging
            input_data = self._prepare_input_log_data()

            self.logger.debug(
                f"Provider: {self.config['llm_provider']}, Model: {self.config['llm_model']}"
            )

            # Make the LLM API call
            response = self._make_llm_call()

            # Process the response
            result = self._process_response(response)

            # Log the interaction if logging is enabled
            self._log_llm_interaction(input_data, result)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"Contextual LLM task '{self.name}' completed successfully in {execution_time:.3f}s"
            )

            return TaskResult.success(
                data=result,
                execution_time=execution_time,
                metadata={
                    "task_name": self.name,
                    "provider": self.config["llm_provider"],
                    "model": self.config["llm_model"],
                    "temperature": self.config.get(
                        "temperature", self.DEFAULT_TEMPERATURE
                    ),
                    "max_tokens": self.config.get(
                        "max_tokens", self.DEFAULT_MAX_TOKENS
                    ),
                },
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"LLMTask run operation failed: Contextual LLM task '{self.name}' failed: {str(e)}"
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_name": self.name,
                    "provider": self.config.get("llm_provider", "unknown"),
                    "model": self.config.get("llm_model", "unknown"),
                    "error_type": type(e).__name__,
                },
            )

    def _resolve_templates(self) -> None:
        """Resolve template strings using context data."""
        # Ensure we always have a context instance, even if empty
        if not self.context_instance:
            from .task_context import TaskContext

            self.context_instance = TaskContext()
            self.logger.debug("Created empty context for template resolution")

        # Resolve prompt template - always use prompt_template (legacy prompt support for backward compatibility)
        prompt_template = self.config.get(
            "prompt_template", self.config.get("prompt", "")
        )
        if prompt_template:
            self.resolved_prompt = self._resolve_template_string(prompt_template)

        # Resolve context template
        context_template = self.config.get("context")
        if context_template and isinstance(context_template, str):
            self.resolved_context = self._resolve_template_string(context_template)

        # Resolve role template
        role_template = self.config.get("role")
        if role_template and isinstance(role_template, str):
            self.resolved_role = self._resolve_template_string(role_template)

    def _resolve_template_string(self, template: str) -> str:
        """Resolve a template string by substituting ${variable} placeholders.

        Args:
            template: Template string with ${variable} placeholders

        Returns:
            Resolved template string
        """
        if not template:
            return template

        # Find all ${variable} patterns
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, template)

        resolved = template
        for match in matches:
            placeholder = "${" + match + "}"

            try:
                # First try to get value from task config (resolved dependencies)
                value = self.config.get(match)

                # If not in config, try to resolve from context as path
                if value is None and self.context_instance:
                    value = self.context_instance.get_path_value(match)

                if value is not None:
                    # Convert value to string representation
                    if isinstance(value, str):
                        replacement = value
                    elif isinstance(value, (int, float, bool)):
                        replacement = str(value)
                    elif isinstance(value, list):
                        replacement = self._format_list_for_prompt(value)  # pyright: ignore
                    elif isinstance(value, dict):
                        replacement = self._format_dict_for_prompt(value)  # pyright: ignore
                    else:
                        replacement = str(value)

                    resolved = resolved.replace(placeholder, replacement)
                else:
                    self.logger.warning(f"Could not resolve template variable: {match}")
                    # Leave placeholder as-is if cannot resolve

            except Exception as e:
                self.logger.error(
                    f"Error resolving template variable '{match}': {str(e)}"
                )
                # Leave placeholder as-is on error

        return resolved

    def _format_list_for_prompt(self, value: List[Any]) -> str:
        """Format a list value for inclusion in prompts.

        Args:
            value: List value to format

        Returns:
            Formatted string representation
        """
        if not value:
            return "[]"

        # If list contains dictionaries, format them nicely
        if all(isinstance(item, dict) for item in value):
            formatted_items: List[str] = []
            for i, item in enumerate(value):
                if isinstance(item, dict) and "title" in item and "content" in item:
                    # Format as article
                    formatted_items.append(
                        f"Article {i + 1}: {item['title']}\n{item['content']}"
                    )
                elif isinstance(item, dict) and "url" in item and "title" in item:
                    # Format as search result
                    formatted_items.append(
                        f"Result {i + 1}: {item['title']} ({item['url']})"
                    )
                else:
                    # Generic dict formatting
                    formatted_items.append(
                        f"Item {i + 1}: {self._format_dict_for_prompt(item)}"  # pyright: ignore
                    )
            return "\n\n".join(formatted_items)

        # If list contains simple values
        elif all(isinstance(item, (str, int, float)) for item in value):
            return ", ".join(str(item) for item in value)

        # Mixed or complex list
        else:
            return "\n".join(f"- {str(item)}" for item in value)

    def _format_dict_for_prompt(self, value: Dict[str, Any]) -> str:
        """Format a dictionary value for inclusion in prompts.

        Args:
            value: Dictionary value to format

        Returns:
            Formatted string representation
        """
        if not value:
            return "{}"

        # Special formatting for common structures
        if "title" in value and "content" in value:
            return f"Title: {value['title']}\nContent: {value['content']}"
        elif "query" in value and "results" in value:
            return f"Search Query: {value['query']}\nResults: {len(value.get('results', []))} items"
        elif "url" in value and "title" in value:
            return f"{value['title']} ({value['url']})"

        # Generic formatting
        formatted_pairs: List[str] = []
        for key, val in value.items():
            if isinstance(val, (str, int, float, bool)):
                formatted_pairs.append(f"{key}: {val}")
            elif isinstance(val, list) and len(val) <= 3:  # pyright: ignore
                formatted_pairs.append(f"{key}: [{', '.join(str(v) for v in val)}]")  # pyright: ignore
            else:
                formatted_pairs.append(f"{key}: {type(val).__name__}")  # pyright: ignore

        return ", ".join(formatted_pairs)

    def get_resolved_prompt(self) -> Optional[str]:
        """Get the resolved prompt after template substitution.

        Returns:
            Resolved prompt string or None if not yet resolved
        """
        return self.resolved_prompt

    def get_resolved_context(self) -> Optional[str]:
        """Get the resolved context after template substitution.

        Returns:
            Resolved context string or None if not yet resolved
        """
        return self.resolved_context

    def get_resolved_role(self) -> Optional[str]:
        """Get the resolved role after template substitution.

        Returns:
            Resolved role string or None if not yet resolved
        """
        return self.resolved_role

    def preview_resolved_templates(self) -> Dict[str, str]:
        """Preview what the templates would resolve to with current context.

        Returns:
            Dictionary with resolved template previews
        """
        # Ensure we have a context instance
        if not self.context_instance:
            from .task_context import TaskContext

            self.context_instance = TaskContext()

        preview: Dict[str, str] = {}

        # Preview prompt
        prompt_template = self.config.get(
            "prompt_template", self.config.get("prompt", "")
        )
        if prompt_template:
            preview["prompt"] = self._resolve_template_string(prompt_template)

        # Preview context
        context_template = self.config.get("context")
        if context_template and isinstance(context_template, str):
            preview["context"] = self._resolve_template_string(context_template)

        # Preview role
        role_template = self.config.get("role")
        if role_template and isinstance(role_template, str):
            preview["role"] = self._resolve_template_string(role_template)

        return preview

    @override
    def __str__(self) -> str:
        """Enhanced string representation including template info."""
        base_str = super().__str__()
        dep_count = len(self.dependencies)
        has_template = bool(self.config.get("prompt_template"))
        # TODO: Replace ContextualLLMTask everywhere in the project with LLMTask
        return f"ContextualLLMTask({base_str}, dependencies={dep_count}, has_template={has_template})"
