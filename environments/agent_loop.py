"""
HermesAgentLoop -- Reusable Multi-Turn Agent Engine

Runs the hermes-agent tool-calling loop using standard OpenAI-spec tool calling.
Works with any server that returns ChatCompletion objects with tool_calls:
    - Phase 1: OpenAI server type (VLLM, SGLang, OpenRouter, OpenAI API)
    - Phase 2: ManagedServer with client-side tool call parser

The loop passes tools= and checks response.choices[0].message.tool_calls,
identical to hermes-agent's run_agent.py. Tool execution is dispatched via
handle_function_call() from model_tools.py.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from model_tools import handle_function_call
from tools.terminal_tool import get_active_env
from tools.tool_result_storage import maybe_persist_tool_result, enforce_turn_budget

# Thread pool for running sync tool calls that internally use asyncio.run()
# (e.g., the Modal/Docker/Daytona terminal backends). Running them in a separate
# thread gives them a clean event loop so they don't deadlock inside Atropos's loop.
# Size must be large enough for concurrent eval tasks (e.g., 89 TB2 tasks all
# making tool calls). Too small = thread pool starvation, tasks queue for minutes.
# Resized at runtime by HermesAgentBaseEnv.__init__ via resize_tool_pool().
_tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=128)


def resize_tool_pool(max_workers: int):
    """
    Replace the global tool executor with a new one of the given size.

    Called by HermesAgentBaseEnv.__init__ based on config.tool_pool_size.
    Safe to call before any tasks are submitted.
    """
    global _tool_executor
    old_executor = _tool_executor
    _tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    old_executor.shutdown(wait=False)
    logger.info("Tool thread pool resized to %d workers", max_workers)

logger = logging.getLogger(__name__)
_SCRATCHPAD_RE = re.compile(
    r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
    flags=re.DOTALL,
)
_XML_TOOL_CALL_RE = re.compile(
    r"<tool>\s*([A-Za-z_][A-Za-z0-9_]*)\s*</tool>\s*"
    r"<args>\s*(\{.*?\})\s*</args>",
    flags=re.DOTALL | re.IGNORECASE,
)
_DSML_INVOKE_RE = re.compile(
    r"<[^<>\s]*invoke\s+name=\"([A-Za-z_][A-Za-z0-9_]*)\"\s*>(.*?)"
    r"</[^<>\s]*invoke>",
    flags=re.DOTALL | re.IGNORECASE,
)
_DSML_PARAMETER_RE = re.compile(
    r"<[^<>\s]*parameter\s+name=\"([A-Za-z_][A-Za-z0-9_]*)\"\s+"
    r"string=\"(true|false)\"\s*>(.*?)</[^<>\s]*parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)
_MARKDOWN_TOOL_CALL_RE = re.compile(
    r"\*\*Calling:\*\*\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*"
    r"```(?:json)?\s*(\{.*?\})\s*```",
    flags=re.DOTALL | re.IGNORECASE,
)
_BARE_TOOL_CALL_START_RE = re.compile(
    r"(?m)(?:^|[\s`])([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_INTENDED_TOOL_USE_RE = re.compile(
    r"(?is)(\*\*Calling:\*\*|"
    r"\b(?:i'll|i will|let me|i'm going to|i am going to)\b"
    r".{0,160}\b(?:inspect|examine|check|look|open|read|reading|search|run|edit|patch|test|start|continue|see)\b)",
)


@dataclass
class ToolError:
    """Record of a tool execution error during the agent loop."""

    turn: int                  # Which turn the error occurred on
    tool_name: str             # Which tool was called
    arguments: str             # The arguments passed (truncated)
    error: str                 # The error message
    tool_result: str           # The raw result returned to the model


@dataclass
class AgentResult:
    """Result of running the agent loop."""

    # Full conversation history in OpenAI message format
    messages: List[Dict[str, Any]]
    # ManagedServer.get_state() if available (Phase 2), None otherwise
    managed_state: Optional[Dict[str, Any]] = None
    # How many LLM calls were made
    turns_used: int = 0
    # True if model stopped calling tools naturally (vs hitting max_turns)
    finished_naturally: bool = False
    # Extracted reasoning content per turn (from PR #297 helpers)
    reasoning_per_turn: List[Optional[str]] = field(default_factory=list)
    # Tool errors encountered during the loop
    tool_errors: List[ToolError] = field(default_factory=list)
    # Why the loop stopped (finished, api_error, empty_response, max_turns)
    termination_reason: Optional[str] = None
    # API error string if the loop ended because the model call failed
    api_error: Optional[str] = None


def _json_safe(value: Any) -> Any:
    """Convert SDK/Pydantic objects to JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    for method in ("model_dump", "dict", "to_dict"):
        if hasattr(value, method):
            try:
                return _json_safe(getattr(value, method)())
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    return str(value)


def _extract_reasoning_from_message(message) -> Optional[str]:
    """
    Extract reasoning content from a ChatCompletion message.

    Handles multiple provider formats:
    1. message.reasoning_content field (some providers)
    2. message.reasoning field (some providers)
    3. message.reasoning_details[].text (OpenRouter style)

    Note: <think> block extraction from content is NOT done here -- that's
    handled by the response already in Phase 1 (server does it) or by
    ManagedServer's patch in Phase 2.

    Args:
        message: The assistant message from ChatCompletion response

    Returns:
        Extracted reasoning text, or None if not found
    """
    # Check reasoning_content field (common across providers)
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        return message.reasoning_content

    # Check reasoning field
    if hasattr(message, "reasoning") and message.reasoning:
        return message.reasoning

    # Check reasoning_details (OpenRouter style)
    if hasattr(message, "reasoning_details") and message.reasoning_details:
        for detail in message.reasoning_details:
            if hasattr(detail, "text") and detail.text:
                return detail.text
            if isinstance(detail, dict) and detail.get("text"):
                return detail["text"]

    return None


def _merge_reasoning_sources(*parts: Optional[str]) -> Optional[str]:
    """Combine reasoning sources without duplicating identical blocks."""
    merged: List[str] = []
    seen: Set[str] = set()
    for part in parts:
        text = (part or "").strip()
        if not text or text in seen:
            continue
        merged.append(text)
        seen.add(text)
    return "\n\n".join(merged) if merged else None


def _extract_scratchpad_from_content(content: str) -> tuple[str, Optional[str]]:
    """Remove <REASONING_SCRATCHPAD> blocks from visible content."""
    if (
        not content
        or "<REASONING_SCRATCHPAD>" not in content
        or "</REASONING_SCRATCHPAD>" not in content
    ):
        return content, None

    scratchpads: List[str] = []

    def _strip(match: re.Match[str]) -> str:
        scratchpad = match.group(1).strip()
        if scratchpad:
            scratchpads.append(scratchpad)
        return ""

    visible = _SCRATCHPAD_RE.sub(_strip, content)
    visible = re.sub(r"\n{3,}", "\n\n", visible).strip()
    reasoning = "\n\n".join(scratchpads) if scratchpads else None
    return visible, reasoning


def _parse_markdown_tool_calls(content: str) -> tuple[str, Optional[List[Dict[str, Any]]]]:
    """Parse Hermes UI-style markdown pseudo tool calls into OpenAI tool_calls."""
    if not content or "**Calling:**" not in content:
        return content, None

    tool_calls: List[Dict[str, Any]] = []
    first_match_start: Optional[int] = None
    for match in _MARKDOWN_TOOL_CALL_RE.finditer(content):
        if first_match_start is None:
            first_match_start = match.start()
        tool_name = match.group(1)
        raw_arguments = match.group(2).strip()
        try:
            parsed_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            continue
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(parsed_arguments, ensure_ascii=False),
                },
            }
        )

    if not tool_calls:
        return content, None
    visible = content[:first_match_start].strip() if first_match_start is not None else ""
    return visible, tool_calls


def _parse_xml_tool_calls(
    content: str,
    valid_tool_names: Set[str],
) -> tuple[str, Optional[List[Dict[str, Any]]]]:
    """Parse custom XML tool calls: <tool>name</tool><args>{...}</args>."""
    if not content or "<tool>" not in content or "<args>" not in content:
        return content, None

    tool_calls: List[Dict[str, Any]] = []
    first_match_start: Optional[int] = None
    for match in _XML_TOOL_CALL_RE.finditer(content):
        tool_name = match.group(1).strip()
        if tool_name not in valid_tool_names:
            continue
        raw_arguments = match.group(2).strip()
        try:
            parsed_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed_arguments, dict):
            continue
        if first_match_start is None:
            first_match_start = match.start()
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(parsed_arguments, ensure_ascii=False),
                },
            }
        )

    if not tool_calls:
        return content, None
    visible = content[:first_match_start].strip() if first_match_start is not None else ""
    return visible, tool_calls


def _parse_dsml_tool_calls(
    content: str,
    valid_tool_names: Set[str],
) -> tuple[str, Optional[List[Dict[str, Any]]]]:
    """Parse DeepSeek DSML tool calls into OpenAI tool_calls."""
    if not content or "invoke name=" not in content:
        return content, None

    tool_calls: List[Dict[str, Any]] = []
    first_match_start: Optional[int] = None
    visible_cut: Optional[int] = None
    for invoke_match in _DSML_INVOKE_RE.finditer(content):
        tool_name = invoke_match.group(1).strip()
        if tool_name not in valid_tool_names:
            continue

        arguments: Dict[str, Any] = {}
        valid_arguments = True
        for param_match in _DSML_PARAMETER_RE.finditer(invoke_match.group(2)):
            param_name = param_match.group(1).strip()
            is_string = param_match.group(2).lower() == "true"
            raw_value = param_match.group(3).strip()
            if is_string:
                arguments[param_name] = raw_value
                continue
            try:
                arguments[param_name] = json.loads(raw_value)
            except json.JSONDecodeError:
                valid_arguments = False
                break

        if not valid_arguments:
            continue
        if first_match_start is None:
            first_match_start = invoke_match.start()
            visible_cut = first_match_start
            tag_start = content.rfind("<", 0, first_match_start)
            if tag_start >= 0 and "tool_calls" in content[tag_start:first_match_start]:
                visible_cut = tag_start
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )

    if not tool_calls:
        return content, None
    visible = content[:visible_cut].strip() if visible_cut is not None else ""
    return visible, tool_calls


def _parse_bare_tool_calls(
    content: str,
    valid_tool_names: Set[str],
) -> tuple[str, Optional[List[Dict[str, Any]]]]:
    """Parse bare function-style pseudo calls such as read_file({"path": "..."})."""
    if not content or not valid_tool_names:
        return content, None

    decoder = json.JSONDecoder()
    tool_calls: List[Dict[str, Any]] = []
    first_match_start: Optional[int] = None

    for match in _BARE_TOOL_CALL_START_RE.finditer(content):
        tool_name = match.group(1)
        if tool_name not in valid_tool_names:
            continue

        args_start = match.end()
        while args_start < len(content) and content[args_start].isspace():
            args_start += 1
        if args_start >= len(content) or content[args_start] != "{":
            continue

        try:
            parsed_arguments, offset = decoder.raw_decode(content[args_start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed_arguments, dict):
            continue

        close_index = args_start + offset
        while close_index < len(content) and content[close_index].isspace():
            close_index += 1
        if close_index >= len(content) or content[close_index] != ")":
            continue

        if first_match_start is None:
            first_match_start = match.start()
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(parsed_arguments, ensure_ascii=False),
                },
            }
        )

    if not tool_calls:
        return content, None
    visible = content[:first_match_start].strip() if first_match_start is not None else ""
    return visible, tool_calls


def _looks_like_intended_tool_use(content: str) -> bool:
    """Return True when a no-tool assistant message is only promising action."""
    return bool(content and _INTENDED_TOOL_USE_RE.search(content))


def _tool_call_trace_dict(tc) -> Dict[str, Any]:
    if isinstance(tc, dict):
        fn = tc.get("function", {})
        return {
            "id": tc.get("id"),
            "name": fn.get("name", tc.get("name")),
            "arguments": fn.get("arguments", tc.get("arguments")),
        }
    return {
        "id": tc.id,
        "name": tc.function.name,
        "arguments": tc.function.arguments,
    }


class HermesAgentLoop:
    """
    Runs hermes-agent's tool-calling loop using standard OpenAI-spec tool calling.

    Same pattern as run_agent.py:
    - Pass tools= to the API
    - Check response.choices[0].message.tool_calls
    - Dispatch via handle_function_call()

    Works identically with any server type -- OpenAI, VLLM, SGLang, OpenRouter,
    or ManagedServer with a parser. The server determines how tool_calls get
    populated on the response.
    """

    def __init__(
        self,
        server,
        tool_schemas: List[Dict[str, Any]],
        valid_tool_names: Set[str],
        max_turns: int = 30,
        task_id: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        budget_config: Optional["BudgetConfig"] = None,
        tool_choice: Optional[Any] = "auto",
        send_tool_schemas: bool = True,
        api_trace_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        request_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        preserve_reasoning_in_history: bool = True,
    ):
        """
        Initialize the agent loop.

        Args:
            server: Server object with chat_completion() method (OpenAIServer,
                    ManagedServer, ServerManager, etc.)
            tool_schemas: OpenAI-format tool definitions from get_tool_definitions()
            valid_tool_names: Set of tool names the model is allowed to call
            max_turns: Maximum number of LLM calls before stopping
            task_id: Unique ID for terminal/browser session isolation
            temperature: Sampling temperature for generation
            max_tokens: Max tokens per generation (None for server default)
            extra_body: Extra parameters passed to the OpenAI client's create() call.
                        Used for OpenRouter provider preferences, transforms, etc.
                        e.g. {"provider": {"ignore": ["DeepInfra"]}}
            budget_config: Tool result persistence budget. Controls per-tool
                        thresholds, per-turn aggregate budget, and preview size.
                        If None, uses DEFAULT_BUDGET (current hardcoded values).
            tool_choice: OpenAI-compatible tool_choice value to send when tools
                        are available. Defaults to "auto" so providers that
                        require an explicit parser trigger produce structured
                        tool_calls instead of plain text.
            send_tool_schemas: Whether to include the OpenAI tools payload in
                        chat requests. Disable for prompt-only/client-side tool
                        protocols such as custom XML tool calls.
            api_trace_callback: Optional callback invoked once per model API call
                        with a JSON-serializable trace record.
            request_transform: Optional callback that can rewrite the outbound
                        chat kwargs before tracing and submission.
            preserve_reasoning_in_history: Whether to append provider-native
                        reasoning_content to assistant messages in chat history.
                        Some templates use this field for prior turns, but custom
                        reasoning endpoints may degrade if hidden reasoning is
                        replayed verbatim.
        """
        from tools.budget_config import DEFAULT_BUDGET
        self.server = server
        self.tool_schemas = tool_schemas
        self.valid_tool_names = valid_tool_names
        self.max_turns = max_turns
        self.task_id = task_id or str(uuid.uuid4())
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self.budget_config = budget_config or DEFAULT_BUDGET
        self.tool_choice = tool_choice
        self.send_tool_schemas = send_tool_schemas
        self.api_trace_callback = api_trace_callback
        self.request_transform = request_transform
        self.preserve_reasoning_in_history = preserve_reasoning_in_history

    def _emit_api_trace(self, record: Dict[str, Any]) -> None:
        """Best-effort callback for per-turn API trace persistence."""
        if not self.api_trace_callback:
            return
        try:
            self.api_trace_callback(record)
        except Exception as e:
            logger.warning("API trace callback failed for task %s: %s", self.task_id[:8], e)

    async def run(self, messages: List[Dict[str, Any]]) -> AgentResult:
        """
        Execute the full agent loop using standard OpenAI tool calling.

        Args:
            messages: Initial conversation messages (system + user).
                      Modified in-place as the conversation progresses.

        Returns:
            AgentResult with full conversation history, managed state, and metadata
        """
        reasoning_per_turn = []
        tool_errors: List[ToolError] = []
        no_tool_retry_count = 0

        # Per-loop TodoStore for the todo tool (ephemeral, dies with the loop)
        from tools.todo_tool import TodoStore, todo_tool as _todo_tool
        _todo_store = TodoStore()

        # Extract user task from first user message for browser_snapshot context
        _user_task = None
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    _user_task = content.strip()[:500]  # Cap to avoid huge strings
                break

        import time as _time

        for turn in range(self.max_turns):
            turn_start = _time.monotonic()

            # Build the chat_completion kwargs
            chat_kwargs = {
                "messages": messages,
                "n": 1,
                "temperature": self.temperature,
            }

            # Only pass tools if we have them
            if self.tool_schemas and self.send_tool_schemas:
                chat_kwargs["tools"] = self.tool_schemas
                if self.tool_choice is not None:
                    chat_kwargs["tool_choice"] = self.tool_choice

            # Only pass max_tokens if explicitly set
            if self.max_tokens is not None:
                chat_kwargs["max_tokens"] = self.max_tokens

            # Inject extra_body for provider-specific params (e.g., OpenRouter
            # provider preferences like banned/preferred providers, transforms)
            if self.extra_body:
                chat_kwargs["extra_body"] = self.extra_body

            if self.request_transform:
                try:
                    transformed = self.request_transform(chat_kwargs)
                    if transformed is not None:
                        chat_kwargs = transformed
                except Exception as e:
                    logger.warning("Request transform failed on turn %d: %s", turn + 1, e)

            trace_record_base = {
                "timestamp": _time.time(),
                "task_id": self.task_id,
                "turn": turn + 1,
                "model": getattr(self.server, "model_name", None),
                "base_url": getattr(self.server, "base_url", None),
                "request": _json_safe(chat_kwargs),
                "tool_schema_names": [
                    schema.get("function", {}).get("name")
                    for schema in self.tool_schemas
                ],
            }

            # Make the API call -- standard OpenAI spec
            api_start = _time.monotonic()
            try:
                response = await self.server.chat_completion(**chat_kwargs)
            except Exception as e:
                api_elapsed = _time.monotonic() - api_start
                self._emit_api_trace(
                    {
                        **trace_record_base,
                        "latency_seconds": api_elapsed,
                        "response_id": None,
                        "response_model": None,
                        "assistant_content": None,
                        "reasoning": None,
                        "tool_calls": [],
                        "finish_reason": None,
                        "usage": None,
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                        },
                    }
                )
                logger.error("API call failed on turn %d (%.1fs): %s", turn + 1, api_elapsed, e)
                return AgentResult(
                    messages=messages,
                    managed_state=self._get_managed_state(),
                    turns_used=turn + 1,
                    finished_naturally=False,
                    reasoning_per_turn=reasoning_per_turn,
                    tool_errors=tool_errors,
                    termination_reason="api_error",
                    api_error=str(e),
                )

            api_elapsed = _time.monotonic() - api_start

            if not response or not response.choices:
                self._emit_api_trace(
                    {
                        **trace_record_base,
                        "latency_seconds": api_elapsed,
                        "response_id": getattr(response, "id", None) if response else None,
                        "response_model": getattr(response, "model", None) if response else None,
                        "assistant_content": None,
                        "reasoning": None,
                        "tool_calls": [],
                        "finish_reason": None,
                        "usage": _json_safe(getattr(response, "usage", None)) if response else None,
                        "error": {
                            "type": "EmptyResponse",
                            "message": "Response missing choices",
                        },
                    }
                )
                logger.warning("Empty response on turn %d (api=%.1fs)", turn + 1, api_elapsed)
                return AgentResult(
                    messages=messages,
                    managed_state=self._get_managed_state(),
                    turns_used=turn + 1,
                    finished_naturally=False,
                    reasoning_per_turn=reasoning_per_turn,
                    tool_errors=tool_errors,
                    termination_reason="empty_response",
                )

            assistant_msg = response.choices[0].message

            raw_assistant_content = assistant_msg.content or ""
            assistant_content = raw_assistant_content
            scratchpad_reasoning = None
            if isinstance(raw_assistant_content, str):
                assistant_content, scratchpad_reasoning = _extract_scratchpad_from_content(
                    raw_assistant_content
                )

            # Extract reasoning content from the response (all provider formats),
            # then merge any inline scratchpad reasoning for traces/metrics.
            native_reasoning = _extract_reasoning_from_message(assistant_msg)
            reasoning = _merge_reasoning_sources(native_reasoning, scratchpad_reasoning)
            reasoning_per_turn.append(reasoning)

            # Check for tool calls -- standard OpenAI spec.
            # Fallback: if response has no structured tool_calls but content
            # contains raw tool call tags (e.g. <tool_call>), parse them using
            # hermes-agent's standalone parsers. This handles the case where
            # ManagedServer's ToolCallTranslator couldn't parse because vLLM
            # isn't installed.
            if (
                not assistant_msg.tool_calls
                and assistant_msg.content
                and self.tool_schemas
                and "<tool_call>" in (assistant_msg.content or "")
            ):
                try:
                    from environments.tool_call_parsers import get_parser
                    fallback_parser = get_parser("hermes")
                    parsed_content, parsed_calls = fallback_parser.parse(
                        assistant_msg.content
                    )
                    if parsed_calls:
                        assistant_msg.tool_calls = parsed_calls
                        if parsed_content is not None:
                            assistant_msg.content = parsed_content
                        logger.debug(
                            "Fallback parser extracted %d tool calls from raw content",
                            len(parsed_calls),
                        )
                except Exception:
                    pass  # Fall through to no tool calls

            if (
                not assistant_msg.tool_calls
                and assistant_msg.content
                and self.tool_schemas
            ):
                parsed_content, parsed_calls = _parse_dsml_tool_calls(
                    assistant_msg.content,
                    self.valid_tool_names,
                )
                if parsed_calls:
                    assistant_msg.tool_calls = parsed_calls
                    assistant_msg.content = parsed_content
                    logger.debug(
                        "DSML tool-call parser extracted %d tool calls from raw content",
                        len(parsed_calls),
                    )

            if (
                not assistant_msg.tool_calls
                and assistant_msg.content
                and self.tool_schemas
            ):
                parsed_content, parsed_calls = _parse_xml_tool_calls(
                    assistant_msg.content,
                    self.valid_tool_names,
                )
                if parsed_calls:
                    assistant_msg.tool_calls = parsed_calls
                    assistant_msg.content = parsed_content
                    logger.debug(
                        "XML tool-call parser extracted %d tool calls from raw content",
                        len(parsed_calls),
                    )

            if (
                not assistant_msg.tool_calls
                and assistant_msg.content
                and self.tool_schemas
                and "**Calling:**" in (assistant_msg.content or "")
            ):
                parsed_content, parsed_calls = _parse_markdown_tool_calls(
                    assistant_msg.content
                )
                if parsed_calls:
                    assistant_msg.tool_calls = parsed_calls
                    assistant_msg.content = parsed_content
                    logger.debug(
                        "Markdown pseudo-call parser extracted %d tool calls from raw content",
                        len(parsed_calls),
                    )

            if (
                not assistant_msg.tool_calls
                and assistant_msg.content
                and self.tool_schemas
            ):
                parsed_content, parsed_calls = _parse_bare_tool_calls(
                    assistant_msg.content,
                    self.valid_tool_names,
                )
                if parsed_calls:
                    assistant_msg.tool_calls = parsed_calls
                    assistant_msg.content = parsed_content
                    logger.debug(
                        "Bare pseudo-call parser extracted %d tool calls from raw content",
                        len(parsed_calls),
                    )

            if (
                not assistant_msg.tool_calls
                and native_reasoning
                and self.tool_schemas
            ):
                parsed_reasoning, parsed_calls = _parse_dsml_tool_calls(
                    native_reasoning,
                    self.valid_tool_names,
                )
                if parsed_calls:
                    assistant_msg.tool_calls = parsed_calls
                    native_reasoning = parsed_reasoning or None
                    reasoning = _merge_reasoning_sources(
                        native_reasoning, scratchpad_reasoning
                    )
                    reasoning_per_turn[-1] = reasoning
                    logger.debug(
                        "DSML tool-call parser extracted %d tool calls from reasoning",
                        len(parsed_calls),
                    )

            # Preserve scratchpad-tagged content verbatim in conversation history so
            # the next assistant turn sees the same raw format it produced.
            history_content = raw_assistant_content if scratchpad_reasoning else (assistant_msg.content or "")

            self._emit_api_trace(
                {
                    **trace_record_base,
                    "latency_seconds": api_elapsed,
                    "response_id": getattr(response, "id", None),
                    "response_model": getattr(response, "model", None),
                    "assistant_content": assistant_content,
                    "assistant_content_raw": (
                        raw_assistant_content
                        if assistant_content != raw_assistant_content
                        else None
                    ),
                    "reasoning": reasoning,
                    "tool_calls": [
                        _tool_call_trace_dict(tc)
                        for tc in (assistant_msg.tool_calls or [])
                    ],
                    "finish_reason": getattr(response.choices[0], "finish_reason", None),
                    "usage": _json_safe(getattr(response, "usage", None)),
                    "error": None,
                }
            )

            if assistant_msg.tool_calls:
                no_tool_retry_count = 0
                # Normalize tool calls to dicts — they may come as objects
                # (OpenAI API) or dicts (vLLM ToolCallTranslator).
                def _tc_to_dict(tc):
                    if isinstance(tc, dict):
                        return {
                            "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                            "type": "function",
                            "function": {
                                "name": tc.get("function", {}).get("name", tc.get("name", "")),
                                "arguments": tc.get("function", {}).get("arguments", tc.get("arguments", "{}")),
                            },
                        }
                    return {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }

                # Build the assistant message dict for conversation history
                msg_dict: Dict[str, Any] = {
                    "role": "assistant",
                    "content": history_content,
                    "tool_calls": [_tc_to_dict(tc) for tc in assistant_msg.tool_calls],
                }

                # Preserve reasoning_content for multi-turn chat template handling
                # (e.g., Kimi-K2's template renders <think> blocks differently
                # for history vs. the latest turn based on this field)
                if native_reasoning and self.preserve_reasoning_in_history:
                    msg_dict["reasoning_content"] = native_reasoning

                messages.append(msg_dict)

                # Execute each tool call via hermes-agent's dispatch
                for tc in assistant_msg.tool_calls:
                    # Handle both object (OpenAI) and dict (vLLM) formats
                    if isinstance(tc, dict):
                        tool_name = tc.get("function", {}).get("name", tc.get("name", ""))
                        tool_args_raw = tc.get("function", {}).get("arguments", tc.get("arguments", "{}"))
                    else:
                        tool_name = tc.function.name
                        tool_args_raw = tc.function.arguments

                    # Validate tool name
                    if tool_name not in self.valid_tool_names:
                        tool_result = json.dumps(
                            {
                                "error": f"Unknown tool '{tool_name}'. "
                                f"Available tools: {sorted(self.valid_tool_names)}"
                            }
                        )
                        tool_errors.append(ToolError(
                            turn=turn + 1, tool_name=tool_name,
                            arguments=tool_args_raw[:200],
                            error=f"Unknown tool '{tool_name}'",
                            tool_result=tool_result,
                        ))
                        logger.warning(
                            "Model called unknown tool '%s' on turn %d",
                            tool_name, turn + 1,
                        )
                    else:
                        # Parse arguments
                        try:
                            args = json.loads(tool_args_raw)
                        except json.JSONDecodeError as e:
                            args = None
                            tool_result = json.dumps(
                                {"error": f"Invalid JSON in tool arguments: {e}. Please retry with valid JSON."}
                            )
                            tool_errors.append(ToolError(
                                turn=turn + 1, tool_name=tool_name,
                                arguments=tool_args_raw[:200],
                                error=f"Invalid JSON: {e}",
                                tool_result=tool_result,
                            ))
                            logger.warning(
                                "Invalid JSON in tool call arguments for '%s': %s",
                                tool_name, tool_args_raw[:200],
                            )

                        # Dispatch tool only if arguments parsed successfully
                        if args is not None:
                            try:
                                if tool_name == "terminal":
                                    backend = os.getenv("TERMINAL_ENV", "local")
                                    cmd_preview = args.get("command", "")[:80]
                                    logger.info(
                                        "[%s] $ %s", self.task_id[:8], cmd_preview,
                                    )

                                tool_submit_time = _time.monotonic()

                                # Todo tool -- handle locally (needs per-loop TodoStore)
                                if tool_name == "todo":
                                    tool_result = _todo_tool(
                                        todos=args.get("todos"),
                                        merge=args.get("merge", False),
                                        store=_todo_store,
                                    )
                                    tool_elapsed = _time.monotonic() - tool_submit_time
                                elif tool_name == "memory":
                                    tool_result = json.dumps({"error": "Memory is not available in RL environments."})
                                    tool_elapsed = _time.monotonic() - tool_submit_time
                                elif tool_name == "session_search":
                                    tool_result = json.dumps({"error": "Session search is not available in RL environments."})
                                    tool_elapsed = _time.monotonic() - tool_submit_time
                                else:
                                    # Run tool calls in a thread pool so backends that
                                    # use asyncio.run() internally (modal, docker, daytona) get
                                    # a clean event loop instead of deadlocking.
                                    loop = asyncio.get_event_loop()
                                    # Capture current tool_name/args for the lambda
                                    _tn, _ta, _tid = tool_name, args, self.task_id
                                    tool_result = await loop.run_in_executor(
                                        _tool_executor,
                                        lambda: handle_function_call(
                                            _tn, _ta, task_id=_tid,
                                            user_task=_user_task,
                                        ),
                                    )
                                    tool_elapsed = _time.monotonic() - tool_submit_time

                                # Log slow tools and thread pool stats for debugging
                                pool_active = _tool_executor._work_queue.qsize()
                                if tool_elapsed > 30:
                                    logger.warning(
                                        "[%s] turn %d: %s took %.1fs (pool queue=%d)",
                                        self.task_id[:8], turn + 1, tool_name,
                                        tool_elapsed, pool_active,
                                    )
                            except Exception as e:
                                tool_result = json.dumps(
                                    {"error": f"Tool execution failed: {type(e).__name__}: {str(e)}"}
                                )
                                tool_errors.append(ToolError(
                                    turn=turn + 1, tool_name=tool_name,
                                    arguments=tool_args_raw[:200],
                                    error=f"{type(e).__name__}: {str(e)}",
                                    tool_result=tool_result,
                                ))
                                logger.error(
                                    "Tool '%s' execution failed on turn %d: %s",
                                    tool_name, turn + 1, e,
                                )

                        # Also check if the tool returned an error in its JSON result
                        try:
                            result_data = json.loads(tool_result)
                            if isinstance(result_data, dict):
                                err = result_data.get("error")
                                exit_code = result_data.get("exit_code")
                                if err and exit_code and exit_code < 0:
                                    tool_errors.append(ToolError(
                                        turn=turn + 1, tool_name=tool_name,
                                        arguments=tool_args_raw[:200],
                                        error=str(err),
                                        tool_result=tool_result[:500],
                                    ))
                        except (json.JSONDecodeError, TypeError):
                            pass

                    tc_id = tc.get("id", "") if isinstance(tc, dict) else tc.id
                    tool_result = maybe_persist_tool_result(
                        content=tool_result,
                        tool_name=tool_name,
                        tool_use_id=tc_id,
                        env=get_active_env(self.task_id),
                        config=self.budget_config,
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": tool_result,
                        }
                    )

                num_tcs = len(assistant_msg.tool_calls)
                if num_tcs > 0:
                    enforce_turn_budget(
                        messages[-num_tcs:],
                        env=get_active_env(self.task_id),
                        config=self.budget_config,
                    )

                turn_elapsed = _time.monotonic() - turn_start
                logger.info(
                    "[%s] turn %d: api=%.1fs, %d tools, turn_total=%.1fs",
                    self.task_id[:8], turn + 1, api_elapsed,
                    len(assistant_msg.tool_calls), turn_elapsed,
                )

            else:
                # No tool calls -- model is done
                msg_dict = {
                    "role": "assistant",
                    "content": history_content,
                }
                if native_reasoning and self.preserve_reasoning_in_history:
                    msg_dict["reasoning_content"] = native_reasoning
                messages.append(msg_dict)

                if (
                    self.tool_schemas
                    and no_tool_retry_count < 2
                    and _looks_like_intended_tool_use(history_content)
                ):
                    no_tool_retry_count += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You described tool use but did not make a structured "
                                "tool call. Make the tool call now using the available "
                                "tools; do not write the call in prose or markdown."
                            ),
                        }
                    )
                    turn_elapsed = _time.monotonic() - turn_start
                    logger.info(
                        "[%s] turn %d: api=%.1fs, intended tool use without tool_calls "
                        "(retry %d), turn_total=%.1fs",
                        self.task_id[:8], turn + 1, api_elapsed,
                        no_tool_retry_count, turn_elapsed,
                    )
                    continue

                turn_elapsed = _time.monotonic() - turn_start
                logger.info(
                    "[%s] turn %d: api=%.1fs, no tools (finished), turn_total=%.1fs",
                    self.task_id[:8], turn + 1, api_elapsed, turn_elapsed,
                )

                return AgentResult(
                    messages=messages,
                    managed_state=self._get_managed_state(),
                    turns_used=turn + 1,
                    finished_naturally=True,
                    reasoning_per_turn=reasoning_per_turn,
                    tool_errors=tool_errors,
                    termination_reason="finished",
                )

        # Hit max turns without the model stopping
        logger.info("Agent hit max_turns (%d) without finishing", self.max_turns)
        return AgentResult(
            messages=messages,
            managed_state=self._get_managed_state(),
            turns_used=self.max_turns,
            finished_naturally=False,
            reasoning_per_turn=reasoning_per_turn,
            tool_errors=tool_errors,
            termination_reason="max_turns",
        )

    def _get_managed_state(self) -> Optional[Dict[str, Any]]:
        """
        Get ManagedServer state if the server supports it.

        Returns state dict with SequenceNodes containing tokens/logprobs/masks,
        or None if the server doesn't support get_state() (e.g., regular OpenAI server).
        """
        if hasattr(self.server, "get_state"):
            return self.server.get_state()
        return None
