#!/usr/bin/env python3
"""Standalone SWE-rebench MVP runner built on HermesAgentLoop."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.prompt_caching import apply_anthropic_cache_control
from environments.agent_loop import AgentResult, HermesAgentLoop
from environments.benchmarks.swe_rebench.dataset import load_rows
from environments.benchmarks.swe_rebench.prompts import DEFAULT_SYSTEM_PROMPT, build_user_prompt
from environments.benchmarks.swe_rebench.scoring import (
    classify_execution_issue,
    parse_log,
    score_parsed_log,
    write_sanitized_model_patch,
)
from environments.tool_context import ToolContext
from model_tools import get_tool_definitions
from tools.terminal_tool import clear_task_env_overrides, cleanup_vm, register_task_env_overrides

logger = logging.getLogger("swe_rebench_mvp")


@dataclass
class RetryableInstanceError(Exception):
    """Exception wrapper used to signal a retryable infrastructure failure."""

    message: str

    def __str__(self) -> str:
        return self.message


class WorkspaceSpendLimitError(RuntimeError):
    """Raised when Modal reports that the workspace budget has been exhausted."""


_WORKSPACE_SPEND_LIMIT_NEEDLE = "workspace has exceeded its spend limit"
_VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
_DSML_TOKEN = "｜DSML｜"
_THINKING_START_TOKEN = "<think>"
_THINKING_END_TOKEN = "</think>"


def _contains_workspace_spend_limit(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_contains_workspace_spend_limit(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_workspace_spend_limit(item) for item in value)
    return _WORKSPACE_SPEND_LIMIT_NEEDLE in str(value).lower()


def _raise_if_workspace_spend_limit(*values: Any, context: str) -> None:
    if _contains_workspace_spend_limit(values):
        raise WorkspaceSpendLimitError(f"Workspace has exceeded its spend limit during {context}")


class OpenAIChatServer:
    """Minimal server wrapper matching HermesAgentLoop.chat_completion expectations."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str,
        timeout_seconds: float,
    ):
        self.base_url = base_url
        self.model_name = model_name
        default_headers: Dict[str, str] = {}
        if "openrouter.ai" in base_url:
            default_headers = {
                "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
                "X-OpenRouter-Title": "Hermes SWE-rebench MVP",
            }
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers or None,
        )
        self.timeout_seconds = timeout_seconds

    async def chat_completion(self, **kwargs):
        return await self.client.chat.completions.create(
            model=self.model_name,
            timeout=self.timeout_seconds,
            **kwargs,
        )


class ApiTraceSink:
    """Stream per-turn API traces while aggregating usage for spend summaries."""

    def __init__(
        self,
        *,
        path: Optional[Path],
        prompt_cost_per_million: float,
        completion_cost_per_million: float,
    ):
        self.path = path
        self.prompt_cost_per_million = prompt_cost_per_million
        self.completion_cost_per_million = completion_cost_per_million
        self.calls = 0
        self.errors = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cached_prompt_tokens = 0
        self.reasoning_tokens = 0
        self.calls_with_reasoning = 0
        self.reasoning_chars = 0
        self.provider_cost_usd = 0.0
        self.provider_prompt_cost_usd = 0.0
        self.provider_completion_cost_usd = 0.0
        self.calls_with_provider_cost = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self.path.unlink()

    def __call__(self, record: Dict[str, Any]) -> None:
        self.calls += 1
        if record.get("error"):
            self.errors += 1

        usage = record.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.cached_prompt_tokens += int(prompt_details.get("cached_tokens") or 0)
        self.reasoning_tokens += int(completion_details.get("reasoning_tokens") or 0)
        reasoning = record.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            self.calls_with_reasoning += 1
            self.reasoning_chars += len(reasoning)
        cost = usage.get("cost")
        if cost is not None:
            try:
                self.provider_cost_usd += float(cost)
                self.calls_with_provider_cost += 1
            except (TypeError, ValueError):
                pass
        cost_details = usage.get("cost_details") or {}
        try:
            self.provider_prompt_cost_usd += float(cost_details.get("upstream_inference_prompt_cost") or 0.0)
            self.provider_completion_cost_usd += float(cost_details.get("upstream_inference_completions_cost") or 0.0)
        except (TypeError, ValueError):
            pass

        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def summary(self) -> Dict[str, Any]:
        estimated_cost = None
        if self.prompt_cost_per_million > 0 or self.completion_cost_per_million > 0:
            estimated_cost = (
                (self.prompt_tokens / 1_000_000.0) * self.prompt_cost_per_million
                + (self.completion_tokens / 1_000_000.0) * self.completion_cost_per_million
            )
        provider_cost = self.provider_cost_usd if self.calls_with_provider_cost > 0 else None
        effective_cost = provider_cost if provider_cost is not None else estimated_cost
        cost_source = None
        if provider_cost is not None:
            cost_source = "provider_usage"
        elif estimated_cost is not None:
            cost_source = "configured_token_rates"
        return {
            "calls": self.calls,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "calls_with_reasoning": self.calls_with_reasoning,
            "reasoning_chars": self.reasoning_chars,
            "provider_cost_usd": provider_cost,
            "provider_prompt_cost_usd": self.provider_prompt_cost_usd if self.calls_with_provider_cost > 0 else None,
            "provider_completion_cost_usd": self.provider_completion_cost_usd if self.calls_with_provider_cost > 0 else None,
            "estimated_cost_usd": estimated_cost,
            "effective_cost_usd": effective_cost,
            "cost_source": cost_source,
            "raw_trace_path": str(self.path) if self.path else None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SWE-rebench MVP benchmark with Hermes.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=_PROJECT_ROOT / "datasets" / "SWE-rebench-V2_train.jsonl",
        help="Local JSONL dataset path.",
    )
    parser.add_argument(
        "--swe-rebench-src",
        type=Path,
        default=_PROJECT_ROOT / "SWE-rebench-V2",
        help="Path to a local SWE-rebench-V2 checkout used for official log parsers.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=_PROJECT_ROOT / "artifact" / "swe_rebench",
        help="Artifact root directory.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=_PROJECT_ROOT / "logs" / "swe_rebench_eval",
        help="Log directory for runner logs.",
    )
    parser.add_argument("--run-id", default=None, help="Explicit run id; defaults to local-YYYYMMDD_HHMMSS.")
    parser.add_argument("--mode", choices=["no_web", "with_web"], default="no_web")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--instance-ids", default=None, help="Comma-separated subset of instance ids.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Skip instances that already have instance_summary.json.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)

    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4")
    parser.add_argument("--api-key", default=None, help="Explicit API key. If omitted, --api-key-env is used.")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--extra-body-json", default=None, help="JSON object forwarded as extra_body.")
    parser.add_argument(
        "--prompt-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Anthropic/OpenRouter prompt caching. Default is auto for Claude via OpenRouter.",
    )
    parser.add_argument("--prompt-cache-ttl", default="5m", help="Anthropic prompt cache TTL when caching is enabled.")

    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional text file whose contents replace --system-prompt.",
    )
    parser.add_argument("--include-pr-description", action="store_true")
    parser.add_argument(
        "--prefill-messages-file",
        type=Path,
        default=None,
        help="Optional JSON file containing a prefilled list of {role, content} messages.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Provider-native reasoning override: one of max, xhigh, high, medium, "
            "low, minimal, none."
        ),
    )
    parser.add_argument(
        "--save-api-traces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist raw per-turn API traces to api_trace.jsonl.",
    )
    parser.add_argument(
        "--send-tool-schemas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Send OpenAI native tool schemas in model requests. Disable for "
            "prompt-only/client-side tool protocols such as custom XML calls."
        ),
    )
    parser.add_argument(
        "--preserve-reasoning-history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replay provider-native reasoning_content on assistant history "
            "messages. Disabled by default for SWE-rebench because some "
            "reasoning endpoints loop when hidden reasoning is fed back."
        ),
    )

    parser.add_argument("--terminal-timeout", type=int, default=300)
    parser.add_argument("--terminal-lifetime", type=int, default=5400)
    parser.add_argument("--scorer-timeout", type=int, default=1800)
    parser.add_argument("--container-cpu", type=float, default=1.0)
    parser.add_argument("--container-memory", type=int, default=5120)
    parser.add_argument("--container-disk", type=int, default=51200)

    parser.add_argument("--prompt-cost-per-million", type=float, default=0.0)
    parser.add_argument("--completion-cost-per-million", type=float, default=0.0)
    parser.add_argument("--modal-cost-per-hour", type=float, default=0.0)
    return parser.parse_args()


def _load_env() -> None:
    env_path = _REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _jsonl_append(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _parse_extra_body(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--extra-body-json must decode to an object")
    return parsed


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt_file is None:
        return args.system_prompt
    return args.system_prompt_file.read_text(encoding="utf-8").strip()


def _render_system_prompt_templates(
    system_prompt: str,
    tool_schemas: List[Dict[str, Any]],
) -> str:
    """Fill lightweight prompt-template placeholders that depend on runtime tools."""
    if not system_prompt:
        return system_prompt
    rendered = system_prompt
    rendered = rendered.replace("{dsml_token}", _DSML_TOKEN)
    rendered = rendered.replace("{thinking_start_token}", _THINKING_START_TOKEN)
    rendered = rendered.replace("{thinking_end_token}", _THINKING_END_TOKEN)
    if "{tool_schemas}" in rendered:
        rendered = rendered.replace(
            "{tool_schemas}",
            json.dumps(tool_schemas, ensure_ascii=False, indent=2),
        )
    return rendered


def _load_prefill_messages(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None:
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in prefill messages file {path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError(f"Prefill messages file must contain a JSON array: {path}")

    messages: List[Dict[str, str]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Prefill message #{index} in {path} must be an object")
        role = str(item.get("role") or "").strip()
        content = item.get("content")
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Prefill message #{index} in {path} has invalid role {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"Prefill message #{index} in {path} must have string content")
        message: Dict[str, str] = {"role": role, "content": content}
        if "name" in item:
            name = item["name"]
            if not isinstance(name, str):
                raise ValueError(f"Prefill message #{index} in {path} has non-string name")
            message["name"] = name
        messages.append(message)
    return messages


def _parse_reasoning_config(effort: Optional[str]) -> Optional[Dict[str, Any]]:
    if effort is None:
        return None
    normalized = effort.strip().lower()
    if not normalized:
        return None
    if normalized not in _VALID_REASONING_EFFORTS:
        valid = ", ".join(sorted(_VALID_REASONING_EFFORTS))
        raise ValueError(f"--reasoning-effort must be one of: {valid}")
    if normalized == "none":
        return {"effort": "none"}
    return {"enabled": True, "effort": normalized}


def _resolve_extra_body(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    extra_body = _parse_extra_body(args.extra_body_json) or {}
    if args.reasoning_config is not None:
        extra_body["reasoning"] = args.reasoning_config
        effort = args.reasoning_config.get("effort")
        if isinstance(effort, str) and effort:
            # vLLM's chat-completions path uses this top-level field to
            # activate template-level thinking for DeepSeek V4.
            extra_body["reasoning_effort"] = effort
    return extra_body or None


def _set_terminal_env(args: argparse.Namespace) -> None:
    os.environ["TERMINAL_ENV"] = "modal"
    os.environ["TERMINAL_TIMEOUT"] = str(args.terminal_timeout)
    os.environ["TERMINAL_LIFETIME_SECONDS"] = str(args.terminal_lifetime)
    os.environ["TERMINAL_CONTAINER_CPU"] = str(args.container_cpu)
    os.environ["TERMINAL_CONTAINER_MEMORY"] = str(args.container_memory)
    os.environ["TERMINAL_CONTAINER_DISK"] = str(args.container_disk)
    os.environ["TERMINAL_CONTAINER_PERSISTENT"] = "true"


def _toolsets_for_mode(mode: str) -> List[str]:
    toolsets = ["terminal", "file", "vision", "todo"]
    if mode == "with_web":
        toolsets.append("web")
    return toolsets


def _prompt_caching_enabled(args: argparse.Namespace) -> bool:
    if args.prompt_caching is not None:
        return args.prompt_caching
    base_url_lower = args.base_url.lower()
    model_lower = args.model.lower()
    return (
        "claude" in model_lower
        and (
            "openrouter" in base_url_lower
            or "inference-api.nousresearch.com" in base_url_lower
        )
    )


def _prompt_cache_native_tool_layout(args: argparse.Namespace) -> bool:
    """Return True when the endpoint accepts cache markers on role=tool messages."""
    return "inference-api.nousresearch.com" in args.base_url.lower()


def _request_transform(args: argparse.Namespace):
    if not _prompt_caching_enabled(args):
        return None

    def _transform(chat_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        transformed = dict(chat_kwargs)
        transformed["messages"] = apply_anthropic_cache_control(
            transformed["messages"],
            cache_ttl=args.prompt_cache_ttl,
            native_anthropic=_prompt_cache_native_tool_layout(args),
        )
        return transformed

    return _transform


def _effective_network_policy(mode: str) -> Dict[str, Any]:
    """Describe the currently enforceable sandbox network policy."""
    if mode == "no_web":
        return {
            "requested": "block outbound sandbox network",
            "effective": "web tools disabled; sandbox network remains enabled",
            "reason": (
                "Current SWE-ReX Modal sandboxes require open ports for the runtime tunnel, "
                "and Modal rejects open ports when block_network=True."
            ),
        }
    return {
        "requested": "full outbound sandbox network access",
        "effective": "full outbound sandbox network access",
        "reason": "",
    }


def _build_run_layout(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> Dict[str, Path]:
    run_id = args.run_id or f"local-{_timestamp()}"
    run_dir = args.artifact_root / args.mode / run_id
    instances_dir = run_dir / "instances"
    log_dir = args.log_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    instances_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "mode": args.mode,
        "dataset_path": str(args.dataset_path),
        "swe_rebench_src": str(args.swe_rebench_src),
        "max_samples": args.max_samples,
        "offset": args.offset,
        "concurrency": args.concurrency,
        "max_retries": args.max_retries,
        "temperature": args.temperature,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "prompt_caching_enabled": _prompt_caching_enabled(args),
        "prompt_cache_ttl": args.prompt_cache_ttl,
        "system_prompt": args.system_prompt,
        "system_prompt_file": str(args.system_prompt_file) if args.system_prompt_file else None,
        "include_pr_description": args.include_pr_description,
        "prefill_messages_file": str(args.prefill_messages_file) if args.prefill_messages_file else None,
        "prefill_messages": getattr(args, "prefill_messages", []),
        "reasoning_config": getattr(args, "reasoning_config", None),
        "save_api_traces": args.save_api_traces,
        "send_tool_schemas": args.send_tool_schemas,
        "toolsets": _toolsets_for_mode(args.mode),
        "network_policy": _effective_network_policy(args.mode),
        "instances": [row["instance_id"] for row in rows],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _json_dump(run_dir / "manifest.json", manifest)
    return {
        "run_dir": run_dir,
        "instances_dir": instances_dir,
        "log_dir": log_dir,
        "summary_jsonl": run_dir / "summary.jsonl",
        "stats_json": run_dir / "stats.json",
        "spend_summary_json": run_dir / "spend_summary.json",
        "correct_ids_txt": run_dir / "correct_instance_ids.txt",
        "wrong_ids_txt": run_dir / "wrong_instance_ids.txt",
    }


def _instance_dir(layout: Dict[str, Path], instance_id: str) -> Path:
    path = layout["instances_dir"] / instance_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _instance_summary_path(instance_dir: Path) -> Path:
    return instance_dir / "instance_summary.json"


def _workspace_tool_used(messages: List[Dict[str, Any]]) -> bool:
    workspace_tools = {"terminal", "process", "read_file", "write_file", "patch", "search_files"}
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            tool_name = tool_call.get("function", {}).get("name")
            if tool_name in workspace_tools:
                return True
    return False


def _normalize_trajectory(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for message in messages:
        entry = {"role": message.get("role")}
        if "content" in message:
            entry["content"] = message.get("content")
        if "reasoning_content" in message:
            entry["reasoning_content"] = message.get("reasoning_content")
        if "tool_calls" in message:
            entry["tool_calls"] = message.get("tool_calls")
        if "tool_call_id" in message:
            entry["tool_call_id"] = message.get("tool_call_id")
        normalized.append(entry)
    return normalized


def _shell(value: str) -> str:
    return shlex.quote(value)


def _cleanup_task_resources(task_id: str) -> None:
    try:
        cleanup_vm(task_id)
    except Exception:
        pass
    clear_task_env_overrides(task_id)


def _download_large_file(
    ctx: ToolContext,
    remote_path: str,
    local_path: Path,
    *,
    chunk_bytes: int = 24000,
) -> Dict[str, Any]:
    def _detail(result: Dict[str, Any]) -> str:
        parts: List[str] = []
        exit_code = result.get("exit_code")
        if exit_code is not None:
            parts.append(f"exit_code={exit_code}")
        error = str(result.get("error") or "").strip()
        output = str(result.get("output") or "").strip()
        if error:
            parts.append(f"error={error[:500]}")
        if output:
            parts.append(f"output={output[:500]}")
        return ", ".join(parts) if parts else "no detail"

    size_result = ctx.terminal(f"wc -c < {_shell(remote_path)}", timeout=60)
    _raise_if_workspace_spend_limit(size_result, context=f"statting remote file {remote_path}")
    if size_result.get("exit_code") != 0:
        return {
            "success": False,
            "error": f"Could not stat remote file: {remote_path} ({_detail(size_result)})",
        }

    try:
        size = int((size_result.get("output") or "0").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"success": False, "error": f"Unexpected size output for {remote_path}: {size_result.get('output')!r}"}

    local_path.parent.mkdir(parents=True, exist_ok=True)
    if size == 0:
        local_path.write_bytes(b"")
        return {"success": True, "bytes": 0}

    chunk_dir = f"/tmp/hermes_bench_chunks_{uuid.uuid4().hex}"
    split_prefix = f"{chunk_dir}/part_"
    split_result = ctx.terminal(
        f"rm -rf {_shell(chunk_dir)} && mkdir -p {_shell(chunk_dir)} && "
        f"split -b {chunk_bytes} -d -a 6 {_shell(remote_path)} {_shell(split_prefix)}",
        timeout=180,
    )
    _raise_if_workspace_spend_limit(split_result, context=f"splitting remote file {remote_path}")
    if split_result.get("exit_code") != 0:
        return {"success": False, "error": f"Chunk split failed: {_detail(split_result)}"}

    listing = ctx.terminal(f"find {_shell(chunk_dir)} -type f | sort", timeout=60)
    _raise_if_workspace_spend_limit(listing, context=f"listing chunks for {remote_path}")
    if listing.get("exit_code") != 0:
        return {"success": False, "error": f"Could not list chunk dir: {_detail(listing)}"}

    chunk_paths = [line.strip() for line in (listing.get("output") or "").splitlines() if line.strip()]
    if not chunk_paths:
        return {"success": False, "error": f"No chunks created for {remote_path}"}

    tmp_dir = local_path.parent / f".{local_path.name}.parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        with local_path.open("wb") as output_handle:
            for remote_chunk in chunk_paths:
                local_chunk = tmp_dir / Path(remote_chunk).name
                result = ctx.download_file(remote_chunk, str(local_chunk))
                _raise_if_workspace_spend_limit(result, context=f"downloading chunk {remote_chunk}")
                if not result.get("success"):
                    return {"success": False, "error": result.get("error", f"Failed to download chunk {remote_chunk}")}
                output_handle.write(local_chunk.read_bytes())
        return {"success": True, "bytes": size}
    finally:
        ctx.terminal(f"rm -rf {_shell(chunk_dir)}", timeout=30)
        for local_chunk in tmp_dir.glob("*"):
            local_chunk.unlink(missing_ok=True)
        tmp_dir.rmdir()


def _modal_overrides_for(mode: str, row: Dict[str, Any], *, persistent: bool) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "modal_image": row["image_name"],
        "cwd": row["repo_workdir"],
        "container_persistent": persistent,
    }
    return overrides


def _extract_patch_and_status(
    *,
    task_id: str,
    row: Dict[str, Any],
    instance_dir: Path,
) -> Dict[str, Any]:
    ctx = ToolContext(task_id)
    remote_dir = f"/tmp/hermes_swe_rebench_{uuid.uuid4().hex}"
    remote_patch = f"{remote_dir}/model_patch.diff"
    remote_status = f"{remote_dir}/git_status.txt"

    try:
        bash_script = (
            f"set -euo pipefail\n"
            f"mkdir -p {_shell(remote_dir)}\n"
            f"cd {_shell(row['repo_workdir'])}\n"
            f"git status --short > {_shell(remote_status)}\n"
            f"git add -A\n"
            f"git diff --cached --binary > {_shell(remote_patch)}\n"
        )
        result = ctx.terminal(f"bash -lc {_shell(bash_script)}", timeout=180)
        _raise_if_workspace_spend_limit(result, context="extracting patch and git status")
        if result.get("exit_code") != 0:
            raise RuntimeError(
                "Patch extraction failed: "
                f"exit_code={result.get('exit_code')} "
                f"error={result.get('error')} "
                f"output={result.get('output', '')}"
            )

        patch_path = instance_dir / "model_patch.diff"
        status_path = instance_dir / "git_status.txt"
        patch_download = _download_large_file(ctx, remote_patch, patch_path)
        if not patch_download.get("success"):
            raise RuntimeError(patch_download.get("error", "Patch download failed"))
        status_download = _download_large_file(ctx, remote_status, status_path)
        if not status_download.get("success"):
            raise RuntimeError(status_download.get("error", "Status download failed"))

        patch_bytes = patch_path.stat().st_size if patch_path.exists() else 0
        return {
            "patch_path": str(patch_path),
            "git_status_path": str(status_path),
            "patch_bytes": patch_bytes,
            "empty_patch": patch_bytes == 0,
        }
    finally:
        ctx.terminal(f"rm -rf {_shell(remote_dir)}", timeout=30)
        ctx.cleanup()


def _write_local_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_background_scorer(
    ctx: ToolContext,
    *,
    launch_command: str,
    remote_exit: str,
    timeout_seconds: int,
    poll_interval_seconds: float = 15.0,
) -> int:
    launch_result = ctx.terminal(launch_command, timeout=30, background=True)
    _raise_if_workspace_spend_limit(launch_result, context="starting scorer process")
    if launch_result.get("exit_code") != 0:
        raise RuntimeError(f"Failed to start scorer process: {launch_result}")
    process_session_id = str(launch_result.get("session_id") or "").strip()
    if not process_session_id:
        raise RuntimeError(f"Scorer process did not return a session_id: {launch_result}")

    status_command = f"if [ -s {_shell(remote_exit)} ]; then cat {_shell(remote_exit)}; else echo __PENDING__; fi"
    deadline = time.monotonic() + timeout_seconds

    while True:
        status_result = ctx.terminal(status_command, timeout=30)
        _raise_if_workspace_spend_limit(status_result, context="polling scorer process")
        if status_result.get("exit_code") != 0:
            raise RuntimeError(f"Failed to poll scorer process: {status_result}")

        output = str(status_result.get("output") or "").strip()
        status_token = output.splitlines()[-1].strip() if output else ""
        if status_token != "__PENDING__":
            try:
                return int(status_token)
            except ValueError as exc:
                raise RuntimeError(f"Unexpected scorer status output: {output!r}") from exc

        poll_raw = ctx.call_tool("process", {"action": "poll", "session_id": process_session_id})
        try:
            poll_result = json.loads(poll_raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Unexpected process poll output: {poll_raw!r}") from exc
        _raise_if_workspace_spend_limit(poll_result, context="polling scorer process registry")

        process_status = poll_result.get("status")
        if process_status == "running":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if poll_interval_seconds > 0:
                time.sleep(min(poll_interval_seconds, remaining))
            continue
        if process_status == "exited":
            return int(poll_result.get("exit_code", 1))
        raise RuntimeError(f"Failed to poll scorer process: {poll_result}")

    timeout_raw = ctx.call_tool("process", {"action": "kill", "session_id": process_session_id})
    try:
        timeout_result = json.loads(timeout_raw)
    except json.JSONDecodeError:
        timeout_result = {"status": "error", "output": timeout_raw}
    _raise_if_workspace_spend_limit(timeout_result, context="terminating timed-out scorer process")
    return 124


def _score_in_fresh_sandbox(
    *,
    mode: str,
    row: Dict[str, Any],
    swe_rebench_src: Path,
    instance_dir: Path,
    scorer_timeout: int,
) -> Dict[str, Any]:
    scorer_task_id = f"{row['instance_id']}-score-{uuid.uuid4().hex[:8]}"
    register_task_env_overrides(
        scorer_task_id,
        _modal_overrides_for(mode, row, persistent=False),
    )
    ctx = ToolContext(scorer_task_id)
    remote_dir = f"/tmp/hermes_swe_rebench_score_{uuid.uuid4().hex}"
    remote_patch = f"{remote_dir}/model_patch.diff"
    remote_test_patch = f"{remote_dir}/test_patch.diff"
    remote_script = f"{remote_dir}/run_score.sh"
    remote_wrapper = f"{remote_dir}/run_score_wrapper.sh"
    remote_log = f"{remote_dir}/test_log.txt"
    remote_exit = f"{remote_dir}/exit_code.txt"

    local_patch = instance_dir / "model_patch.diff"
    local_sanitized_patch = instance_dir / "model_patch.sanitized.diff"
    local_test_patch = instance_dir / ".test_patch.diff"
    _write_local_text(local_test_patch, str(row.get("test_patch", "")))
    patch_sanitization = write_sanitized_model_patch(
        local_patch,
        str(row.get("test_patch", "")),
        output_path=local_sanitized_patch,
    )

    try:
        mkdir_result = ctx.terminal(f"mkdir -p {_shell(remote_dir)}", timeout=60)
        _raise_if_workspace_spend_limit(mkdir_result, context="creating scorer scratch directory")
        upload_patch = ctx.upload_file(str(local_sanitized_patch), remote_patch)
        _raise_if_workspace_spend_limit(upload_patch, context="uploading model patch")
        if upload_patch.get("exit_code", 0) != 0:
            raise RuntimeError(f"Model patch upload failed: {upload_patch}")
        upload_test_patch = ctx.upload_file(str(local_test_patch), remote_test_patch)
        _raise_if_workspace_spend_limit(upload_test_patch, context="uploading test patch")
        if upload_test_patch.get("exit_code", 0) != 0:
            raise RuntimeError(f"Test patch upload failed: {upload_test_patch}")

        base_commit = str(row.get("base_commit", "")).strip()
        script_lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            f"cd {_shell(row['repo_workdir'])}",
            'actual_head="$(git rev-parse HEAD)"',
            f'if [ "$actual_head" != {_shell(base_commit)} ]; then echo "BASE_COMMIT_MISMATCH expected {base_commit} got $actual_head"; exit 90; fi',
            "git reset --hard HEAD",
            f"if [ -s {_shell(remote_patch)} ]; then git apply -v --3way --recount --ignore-space-change --whitespace=nowarn {_shell(remote_patch)}; fi",
            f"if [ -s {_shell(remote_test_patch)} ]; then git apply -v --3way --recount --ignore-space-change --whitespace=nowarn {_shell(remote_test_patch)}; fi",
        ]
        script_lines.extend(row["install_config"]["test_cmd"])
        write_result = ctx.write_file(remote_script, "\n".join(script_lines) + "\n")
        _raise_if_workspace_spend_limit(write_result, context="writing scorer script")
        if write_result.get("error"):
            raise RuntimeError(f"Failed to write scorer script: {write_result['error']}")

        wrapper_lines = [
            "#!/bin/bash",
            "set -uo pipefail",
            f"bash {_shell(remote_script)} > {_shell(remote_log)} 2>&1",
            "status=$?",
            f"printf '%s\\n' \"$status\" > {_shell(remote_exit)}",
            'exit "$status"',
        ]
        wrapper_result = ctx.write_file(remote_wrapper, "\n".join(wrapper_lines) + "\n")
        _raise_if_workspace_spend_limit(wrapper_result, context="writing scorer wrapper")
        if wrapper_result.get("error"):
            raise RuntimeError(f"Failed to write scorer wrapper: {wrapper_result['error']}")

        chmod_result = ctx.terminal(
            f"chmod +x {_shell(remote_script)} {_shell(remote_wrapper)}",
            timeout=30,
        )
        _raise_if_workspace_spend_limit(chmod_result, context="chmod on scorer script")
        if chmod_result.get("exit_code", 0) != 0:
            raise RuntimeError(f"Failed to chmod scorer script: {chmod_result}")

        launch_command = f"bash {_shell(remote_wrapper)}"
        exit_code = _run_background_scorer(
            ctx,
            launch_command=launch_command,
            remote_exit=remote_exit,
            timeout_seconds=scorer_timeout,
        )
        log_path = instance_dir / "test_log.txt"
        log_download = _download_large_file(ctx, remote_log, log_path)
        if not log_download.get("success"):
            raise RuntimeError(log_download.get("error", "Failed to download scorer log"))
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_log(swe_rebench_src, row["install_config"]["log_parser"], log_text)
        score = score_parsed_log(
            row,
            parsed,
            exit_code=exit_code,
            log_path=str(log_path),
            log_text=log_text,
        )
        score["patch_path"] = str(local_sanitized_patch)
        score["original_patch_path"] = str(local_patch)
        score["sanitized_patch_changed"] = patch_sanitization["changed"]
        score["sanitized_patch_removed_files"] = patch_sanitization["removed_files"]
        score["sanitized_patch_kept_files"] = patch_sanitization["kept_files"]
        return score
    finally:
        ctx.terminal(f"rm -rf {_shell(remote_dir)}", timeout=30)
        ctx.cleanup()
        clear_task_env_overrides(scorer_task_id)
        local_test_patch.unlink(missing_ok=True)


async def _run_rollout(
    *,
    args: argparse.Namespace,
    server: OpenAIChatServer,
    row: Dict[str, Any],
    tool_schemas: List[Dict[str, Any]],
    valid_tool_names: set[str],
    instance_dir: Path,
) -> Dict[str, Any]:
    rollout_task_id = f"{row['instance_id']}-rollout-{uuid.uuid4().hex[:8]}"
    register_task_env_overrides(
        rollout_task_id,
        # The benchmark extracts/scored the patch while the rollout sandbox is still
        # live, so snapshot persistence is unnecessary here and only burns Modal
        # workspace budget.
        _modal_overrides_for(args.mode, row, persistent=False),
    )

    prompt_path = instance_dir / "prompt.txt"
    messages_path = instance_dir / "messages.json"
    trajectory_path = instance_dir / "trajectory.json"
    api_trace_path = instance_dir / "api_trace.jsonl" if args.save_api_traces else None
    prompt = build_user_prompt(row, include_pr_description=args.include_pr_description)
    prompt_path.write_text(prompt, encoding="utf-8")

    messages: List[Dict[str, Any]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    for prefill in getattr(args, "prefill_messages", []):
        messages.append(dict(prefill))
    messages.append({"role": "user", "content": prompt})

    trace_sink = ApiTraceSink(
        path=api_trace_path,
        prompt_cost_per_million=args.prompt_cost_per_million,
        completion_cost_per_million=args.completion_cost_per_million,
    )

    rollout_start = time.monotonic()
    try:
        agent = HermesAgentLoop(
            server=server,
            tool_schemas=tool_schemas,
            valid_tool_names=valid_tool_names,
            max_turns=args.max_turns,
            task_id=rollout_task_id,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body=args.extra_body,
            send_tool_schemas=args.send_tool_schemas,
            api_trace_callback=trace_sink,
            request_transform=_request_transform(args),
            preserve_reasoning_in_history=args.preserve_reasoning_history,
        )
        result = await agent.run(messages)
        rollout_seconds = time.monotonic() - rollout_start

        _json_dump(messages_path, result.messages)
        _json_dump(trajectory_path, _normalize_trajectory(result.messages))

        retryable = result.termination_reason in {"api_error", "empty_response"}
        if retryable:
            raise RetryableInstanceError(
                f"rollout terminated with {result.termination_reason}: {result.api_error or 'no detail'}"
            )

        if _workspace_tool_used(result.messages):
            patch_info = await asyncio.to_thread(
                _extract_patch_and_status,
                task_id=rollout_task_id,
                row=row,
                instance_dir=instance_dir,
            )
        else:
            patch_path = instance_dir / "model_patch.diff"
            status_path = instance_dir / "git_status.txt"
            patch_path.write_text("", encoding="utf-8")
            status_path.write_text("", encoding="utf-8")
            patch_info = {
                "patch_path": str(patch_path),
                "git_status_path": str(status_path),
                "patch_bytes": 0,
                "empty_patch": True,
            }

        rollout_meta = {
            "instance_id": row["instance_id"],
            "mode": args.mode,
            "rollout_task_id": rollout_task_id,
            "finished_naturally": result.finished_naturally,
            "termination_reason": result.termination_reason,
            "api_error": result.api_error,
            "turns_used": result.turns_used,
            "tool_errors": [error.__dict__ for error in result.tool_errors],
            "rollout_seconds": rollout_seconds,
            "messages_path": str(messages_path),
            "trajectory_path": str(trajectory_path),
            "api_trace_path": str(api_trace_path) if api_trace_path else None,
            **patch_info,
        }
        _json_dump(instance_dir / "rollout_meta.json", rollout_meta)
        return {
            "agent_result": result,
            "rollout_meta": rollout_meta,
            "trace_summary": trace_sink.summary(),
        }
    finally:
        _cleanup_task_resources(rollout_task_id)


def _estimate_modal_cost(seconds: float, modal_cost_per_hour: float) -> Optional[float]:
    if modal_cost_per_hour <= 0:
        return None
    return (seconds / 3600.0) * modal_cost_per_hour


def _build_error_summary(
    *,
    args: argparse.Namespace,
    row: Dict[str, Any],
    instance_dir: Path,
    attempt: int,
    error: str,
) -> Dict[str, Any]:
    issue = classify_execution_issue(error=error, exit_code=None)
    return {
        "instance_id": row["instance_id"],
        "mode": args.mode,
        "attempt": attempt,
        "correct": False,
        "scorable": False,
        "failure_category": "runner_error" if issue is None else issue["category"],
        "failure_reason": error if issue is None else issue["reason"],
        "exit_code": None,
        "from_fail_to_pass": [],
        "failed_from_pass_to_pass": [],
        "passed_actual": [],
        "failed_actual": [],
        "error": error,
        "log_parser": row["install_config"]["log_parser"],
        "test_cmd": row["install_config"]["test_cmd"],
        "log_path": None,
        "patch_path": str((instance_dir / "model_patch.diff")),
        "trajectory_path": str((instance_dir / "trajectory.json")),
        "api_trace_path": str((instance_dir / "api_trace.jsonl")) if args.save_api_traces else None,
        "api_prompt_tokens": 0,
        "api_completion_tokens": 0,
        "api_total_tokens": 0,
        "api_cached_prompt_tokens": 0,
        "api_provider_cost_usd": None,
        "api_cost_usd": None,
        "api_cost_source": None,
        "api_estimated_cost_usd": None,
        "modal_estimated_cost_usd": None,
        "total_cost_usd": None,
        "total_estimated_cost_usd": None,
        "rollout_seconds": 0.0,
        "scorer_seconds": 0.0,
        "artifact_dir": str(instance_dir),
    }


async def _evaluate_instance_once(
    *,
    args: argparse.Namespace,
    server: OpenAIChatServer,
    row: Dict[str, Any],
    tool_schemas: List[Dict[str, Any]],
    valid_tool_names: set[str],
    instance_dir: Path,
    attempt: int,
) -> Dict[str, Any]:
    _json_dump(instance_dir / "input.json", row)
    rollout = await _run_rollout(
        args=args,
        server=server,
        row=row,
        tool_schemas=tool_schemas,
        valid_tool_names=valid_tool_names,
        instance_dir=instance_dir,
    )
    scoring_start = time.monotonic()
    score = await asyncio.to_thread(
        _score_in_fresh_sandbox,
        mode=args.mode,
        row=row,
        swe_rebench_src=args.swe_rebench_src,
        instance_dir=instance_dir,
        scorer_timeout=args.scorer_timeout,
    )
    scoring_seconds = time.monotonic() - scoring_start

    usage = rollout["trace_summary"]
    modal_rollout_seconds = rollout["rollout_meta"]["rollout_seconds"]
    modal_scorer_seconds = scoring_seconds
    modal_total_seconds = modal_rollout_seconds + modal_scorer_seconds

    usage_payload = {
        "instance_id": row["instance_id"],
        "api": usage,
        "prompt_caching": {
            "enabled": _prompt_caching_enabled(args),
            "ttl": args.prompt_cache_ttl if _prompt_caching_enabled(args) else None,
        },
        "modal": {
            "rollout_seconds": modal_rollout_seconds,
            "scorer_seconds": modal_scorer_seconds,
            "total_seconds": modal_total_seconds,
        },
    }
    spend_payload = {
        "instance_id": row["instance_id"],
        "api_cost_usd": usage.get("effective_cost_usd"),
        "api_cost_source": usage.get("cost_source"),
        "api_provider_cost_usd": usage.get("provider_cost_usd"),
        "api_estimated_cost_usd": usage.get("estimated_cost_usd"),
        "modal_estimated_cost_usd": _estimate_modal_cost(modal_total_seconds, args.modal_cost_per_hour),
    }
    if spend_payload["api_cost_usd"] is not None and spend_payload["modal_estimated_cost_usd"] is not None:
        spend_payload["total_cost_usd"] = (
            spend_payload["api_cost_usd"] + spend_payload["modal_estimated_cost_usd"]
        )
    else:
        spend_payload["total_cost_usd"] = None
    spend_payload["total_estimated_cost_usd"] = spend_payload["total_cost_usd"]

    _json_dump(instance_dir / "usage.json", usage_payload)
    _json_dump(instance_dir / "spend.json", spend_payload)
    _json_dump(instance_dir / "score.json", score)

    summary = {
        "instance_id": row["instance_id"],
        "mode": args.mode,
        "attempt": attempt,
        "correct": score.get("correct", False),
        "scorable": score.get("scorable", True),
        "failure_category": score.get("failure_category"),
        "failure_reason": score.get("failure_reason", ""),
        "exit_code": score.get("exit_code"),
        "from_fail_to_pass": score.get("from_fail_to_pass", []),
        "failed_from_pass_to_pass": score.get("failed_from_pass_to_pass", []),
        "passed_actual": score.get("passed_actual", []),
        "failed_actual": score.get("failed_actual", []),
        "error": score.get("error", ""),
        "log_parser": row["install_config"]["log_parser"],
        "test_cmd": row["install_config"]["test_cmd"],
        "log_path": score.get("log_path"),
        "patch_path": rollout["rollout_meta"]["patch_path"],
        "trajectory_path": rollout["rollout_meta"]["trajectory_path"],
        "api_trace_path": usage.get("raw_trace_path"),
        "api_prompt_tokens": usage.get("prompt_tokens"),
        "api_completion_tokens": usage.get("completion_tokens"),
        "api_total_tokens": usage.get("total_tokens"),
        "api_cached_prompt_tokens": usage.get("cached_prompt_tokens"),
        "api_provider_cost_usd": spend_payload["api_provider_cost_usd"],
        "api_cost_usd": spend_payload["api_cost_usd"],
        "api_cost_source": spend_payload["api_cost_source"],
        "api_estimated_cost_usd": spend_payload["api_estimated_cost_usd"],
        "modal_estimated_cost_usd": spend_payload["modal_estimated_cost_usd"],
        "total_cost_usd": spend_payload["total_cost_usd"],
        "total_estimated_cost_usd": spend_payload["total_estimated_cost_usd"],
        "rollout_seconds": modal_rollout_seconds,
        "scorer_seconds": modal_scorer_seconds,
        "artifact_dir": str(instance_dir),
    }
    _json_dump(_instance_summary_path(instance_dir), summary)
    return summary


async def _evaluate_instance(
    *,
    args: argparse.Namespace,
    server: OpenAIChatServer,
    row: Dict[str, Any],
    tool_schemas: List[Dict[str, Any]],
    valid_tool_names: set[str],
    layout: Dict[str, Path],
) -> Dict[str, Any]:
    instance_dir = _instance_dir(layout, row["instance_id"])
    summary_path = _instance_summary_path(instance_dir)
    if args.resume and summary_path.exists():
        logger.info("Skipping %s (resume hit)", row["instance_id"])
        return json.loads(summary_path.read_text(encoding="utf-8"))

    last_error: Optional[Exception] = None

    def _non_retryable(exc: Exception) -> bool:
        message = str(exc)
        return _contains_workspace_spend_limit(message) or any(
            needle in message
            for needle in (
                "BASE_COMMIT_MISMATCH",
                "Unknown log parser",
            )
        )

    for attempt in range(1, args.max_retries + 1):
        try:
            logger.info("Evaluating %s (attempt %d/%d)", row["instance_id"], attempt, args.max_retries)
            return await _evaluate_instance_once(
                args=args,
                server=server,
                row=row,
                tool_schemas=tool_schemas,
                valid_tool_names=valid_tool_names,
                instance_dir=instance_dir,
                attempt=attempt,
            )
        except RetryableInstanceError as exc:
            last_error = exc
            logger.warning("Retryable failure for %s: %s", row["instance_id"], exc)
        except Exception as exc:
            last_error = exc
            logger.exception("Infrastructure failure for %s on attempt %d", row["instance_id"], attempt)
            if _contains_workspace_spend_limit(exc):
                error_summary = _build_error_summary(
                    args=args,
                    row=row,
                    instance_dir=instance_dir,
                    attempt=attempt,
                    error=str(exc),
                )
                _json_dump(_instance_summary_path(instance_dir), error_summary)
                raise WorkspaceSpendLimitError(f"{row['instance_id']}: {exc}") from exc
            if _non_retryable(exc):
                logger.error("Non-retryable infrastructure failure for %s: %s", row["instance_id"], exc)
                break

        if attempt < args.max_retries:
            backoff = args.retry_base_seconds * (2 ** (attempt - 1))
            await asyncio.sleep(backoff)

    error_summary = _build_error_summary(
        args=args,
        row=row,
        instance_dir=instance_dir,
        attempt=args.max_retries,
        error=str(last_error) if last_error else "unknown_failure",
    )
    _json_dump(_instance_summary_path(instance_dir), error_summary)
    return error_summary


async def _run_all(args: argparse.Namespace) -> int:
    _load_env()
    _set_terminal_env(args)
    args.system_prompt = _resolve_system_prompt(args)
    args.prefill_messages = _load_prefill_messages(args.prefill_messages_file)
    args.reasoning_config = _parse_reasoning_config(args.reasoning_effort)
    args.extra_body = _resolve_extra_body(args)

    api_key = args.api_key or os.getenv(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"Missing API key. Pass --api-key or set {args.api_key_env}.")

    instance_ids = None
    if args.instance_ids:
        instance_ids = [value.strip() for value in args.instance_ids.split(",") if value.strip()]

    rows = load_rows(
        args.dataset_path,
        offset=args.offset,
        max_samples=args.max_samples,
        instance_ids=instance_ids,
    )
    if not rows:
        raise ValueError("No dataset rows matched the requested slice.")

    requested_toolsets = _toolsets_for_mode(args.mode)
    tool_schemas = get_tool_definitions(enabled_toolsets=requested_toolsets, quiet_mode=True)
    valid_tool_names = {schema["function"]["name"] for schema in tool_schemas}
    required_tools = {"terminal", "read_file", "write_file", "patch"}
    missing_required = sorted(required_tools.difference(valid_tool_names))
    if missing_required:
        raise RuntimeError(f"Missing required tools after resolution: {missing_required}")
    if args.mode == "with_web" and "web_search" not in valid_tool_names:
        raise RuntimeError("with_web mode requested, but the web toolset is unavailable.")

    args.system_prompt = _render_system_prompt_templates(args.system_prompt, tool_schemas)

    layout = _build_run_layout(args, rows)
    _configure_logging(layout["log_dir"] / "runner.log")

    server = OpenAIChatServer(
        base_url=args.base_url,
        model_name=args.model,
        api_key=api_key,
        timeout_seconds=args.request_timeout_seconds,
    )

    logger.info(
        "Starting SWE-rebench MVP run: mode=%s samples=%d concurrency=%d model=%s",
        args.mode,
        len(rows),
        args.concurrency,
        args.model,
    )
    if args.mode == "no_web":
        logger.warning(
            "no_web currently disables web tools but does not block sandbox network: "
            "Modal rejects block_network=True when SWE-ReX opens runtime ports."
        )

    semaphore = asyncio.Semaphore(args.concurrency)
    summaries: List[Dict[str, Any]] = []
    fatal_error: Optional[WorkspaceSpendLimitError] = None

    async def _wrapped(row: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            return await _evaluate_instance(
                args=args,
                server=server,
                row=row,
                tool_schemas=tool_schemas,
                valid_tool_names=valid_tool_names,
                layout=layout,
            )

    tasks = [asyncio.create_task(_wrapped(row)) for row in rows]
    completed = 0
    correct = 0
    for future in asyncio.as_completed(tasks):
        try:
            summary = await future
        except WorkspaceSpendLimitError as exc:
            fatal_error = exc
            logger.error("Aborting run because Modal workspace spend limit was reached: %s", exc)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            break
        summaries.append(summary)
        completed += 1
        if summary.get("correct"):
            correct += 1
        logger.info("Progress %d/%d correct=%d", completed, len(rows), correct)

    summaries = sorted(summaries, key=lambda item: item["instance_id"])
    if layout["summary_jsonl"].exists():
        layout["summary_jsonl"].unlink()
    for summary in summaries:
        _jsonl_append(layout["summary_jsonl"], summary)

    correct_ids = [summary["instance_id"] for summary in summaries if summary.get("correct")]
    wrong_ids = [summary["instance_id"] for summary in summaries if not summary.get("correct")]
    unscorable_ids = [summary["instance_id"] for summary in summaries if summary.get("scorable") is False]
    scorable_summaries = [summary for summary in summaries if summary.get("scorable") is not False]
    scorable_correct_ids = [summary["instance_id"] for summary in scorable_summaries if summary.get("correct")]
    scorable_wrong_ids = [summary["instance_id"] for summary in scorable_summaries if not summary.get("correct")]
    layout["correct_ids_txt"].write_text("\n".join(correct_ids) + ("\n" if correct_ids else ""), encoding="utf-8")
    layout["wrong_ids_txt"].write_text("\n".join(wrong_ids) + ("\n" if wrong_ids else ""), encoding="utf-8")

    total_prompt_tokens = sum(int(summary.get("api_prompt_tokens") or 0) for summary in summaries)
    total_completion_tokens = sum(int(summary.get("api_completion_tokens") or 0) for summary in summaries)
    total_tokens = sum(int(summary.get("api_total_tokens") or 0) for summary in summaries)
    api_costs = [summary.get("api_estimated_cost_usd") for summary in summaries if summary.get("api_estimated_cost_usd") is not None]
    modal_costs = [summary.get("modal_estimated_cost_usd") for summary in summaries if summary.get("modal_estimated_cost_usd") is not None]
    total_costs = [summary.get("total_estimated_cost_usd") for summary in summaries if summary.get("total_estimated_cost_usd") is not None]

    stats = {
        "run_id": layout["run_dir"].name,
        "mode": args.mode,
        "aborted": fatal_error is not None,
        "abort_reason": str(fatal_error) if fatal_error else None,
        "total_instances": len(summaries),
        "correct": len(correct_ids),
        "wrong": len(wrong_ids),
        "accuracy": (len(correct_ids) / len(summaries)) if summaries else 0.0,
        "unscorable": len(unscorable_ids),
        "scorable_instances": len(scorable_summaries),
        "scorable_correct": len(scorable_correct_ids),
        "scorable_wrong": len(scorable_wrong_ids),
        "scorable_accuracy": (len(scorable_correct_ids) / len(scorable_summaries)) if scorable_summaries else 0.0,
        "api_prompt_tokens": total_prompt_tokens,
        "api_completion_tokens": total_completion_tokens,
        "api_total_tokens": total_tokens,
        "api_cached_prompt_tokens": sum(int(summary.get("api_cached_prompt_tokens") or 0) for summary in summaries),
    }
    api_provider_costs = [summary.get("api_provider_cost_usd") for summary in summaries if summary.get("api_provider_cost_usd") is not None]
    api_effective_costs = [summary.get("api_cost_usd") for summary in summaries if summary.get("api_cost_usd") is not None]
    spend_summary = {
        "run_id": layout["run_dir"].name,
        "mode": args.mode,
        "aborted": fatal_error is not None,
        "abort_reason": str(fatal_error) if fatal_error else None,
        "instances": len(summaries),
        "api_prompt_tokens": total_prompt_tokens,
        "api_completion_tokens": total_completion_tokens,
        "api_total_tokens": total_tokens,
        "api_cached_prompt_tokens": stats["api_cached_prompt_tokens"],
        "api_provider_cost_usd": sum(api_provider_costs) if api_provider_costs else None,
        "api_cost_usd": sum(api_effective_costs) if api_effective_costs else None,
        "api_estimated_cost_usd": sum(api_costs) if api_costs else None,
        "modal_estimated_cost_usd": sum(modal_costs) if modal_costs else None,
        "total_cost_usd": sum(total_costs) if total_costs else None,
        "total_estimated_cost_usd": sum(total_costs) if total_costs else None,
        "prompt_caching_enabled": _prompt_caching_enabled(args),
        "prompt_cache_ttl": args.prompt_cache_ttl if _prompt_caching_enabled(args) else None,
    }
    _json_dump(layout["stats_json"], stats)
    _json_dump(layout["spend_summary_json"], spend_summary)

    if fatal_error is not None:
        logger.error(
            "Aborted run %s after Modal workspace spend limit: correct=%d wrong=%d accuracy=%.4f",
            layout["run_dir"].name,
            len(correct_ids),
            len(wrong_ids),
            stats["accuracy"],
        )
        logger.info("Artifacts: %s", layout["run_dir"])
        return 2

    logger.info(
        "Completed run %s: correct=%d wrong=%d accuracy=%.4f",
        layout["run_dir"].name,
        len(correct_ids),
        len(wrong_ids),
        stats["accuracy"],
    )
    logger.info("Artifacts: %s", layout["run_dir"])
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(_run_all(args))


if __name__ == "__main__":
    raise SystemExit(main())
