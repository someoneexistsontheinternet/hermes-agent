"""Prompt construction for SWE-rebench MVP."""

from __future__ import annotations

from typing import Any, Dict

from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    HERMES_AGENT_HELP_GUIDANCE,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
)

BENCHMARK_SYSTEM_PROMPT = (
    "You are a software engineer working inside a repository checkout. "
    "Use the available tools to inspect the codebase, edit files, and verify your fix. "
    "Do not guess: investigate, make the smallest correct change, and stop once the issue is fixed."
)


def build_system_prompt(benchmark_instructions: str = BENCHMARK_SYSTEM_PROMPT) -> str:
    """Build the default Hermes-style system prompt for SWE-rebench rollouts."""
    parts = [
        DEFAULT_AGENT_IDENTITY,
        HERMES_AGENT_HELP_GUIDANCE,
        TOOL_USE_ENFORCEMENT_GUIDANCE,
    ]
    benchmark_instructions = benchmark_instructions.strip()
    if benchmark_instructions:
        parts.append(benchmark_instructions)
    return "\n\n".join(parts)


DEFAULT_SYSTEM_PROMPT = build_system_prompt()


def _meaningful_interface(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return False
    return lowered not in {
        "none",
        "no new interfaces are introduced.",
    }


def build_user_prompt(row: Dict[str, Any], *, include_pr_description: bool = False) -> str:
    """Construct the user-visible prompt from allowed dataset fields only."""
    repo = row["repo"]
    workdir = row["repo_workdir"]
    problem_statement = str(row.get("problem_statement", "")).strip()
    interface = str(row.get("interface", "")).strip()
    pr_description = str(row.get("pr_description", "")).strip()

    sections = [
        f"Repository: {repo}",
        f"Working directory: {workdir}",
        "",
        "Task:",
        problem_statement,
    ]

    if _meaningful_interface(interface):
        sections.extend(["", "Interface constraints:", interface])

    if include_pr_description and pr_description:
        sections.extend(["", "PR description:", pr_description])

    sections.extend(
        [
            "",
            "Instructions:",
            "- Work only inside the repository checkout.",
            "- Use the available tools to inspect, edit, and verify the code.",
            "- When you are done, stop and let the diff be the submission.",
        ]
    )
    return "\n".join(sections).strip()
