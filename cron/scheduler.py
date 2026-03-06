"""
Cron job scheduler - executes due jobs.

Provides tick() which checks for due jobs and runs them. The gateway
calls this every 60 seconds from a background thread.

Uses a file-based lock (~/.hermes/cron/.tick.lock) so only one tick
runs at a time if multiple processes overlap.
"""

import asyncio
import json
import logging
import os
import sys
import traceback

# fcntl is Unix-only; on Windows use msvcrt for file locking
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cron.jobs import get_due_jobs, mark_job_run, save_job_output
from model_runtime_config import load_model_runtime_config

# Resolve Hermes home directory (respects HERMES_HOME override)
_hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))

# File-based lock prevents concurrent ticks from gateway + daemon + systemd timer
_LOCK_DIR = _hermes_home / "cron"
_LOCK_FILE = _LOCK_DIR / ".tick.lock"
_CONTEXT_FILES = ("AGENTS.md", "SOUL.md", ".cursorrules")
_SESSION_ENV_MAP = {
    "platform": "HERMES_SESSION_PLATFORM",
    "chat_id": "HERMES_SESSION_CHAT_ID",
    "chat_type": "HERMES_SESSION_CHAT_TYPE",
    "chat_name": "HERMES_SESSION_CHAT_NAME",
    "thread_id": "HERMES_SESSION_THREAD_ID",
}
_SESSION_ENV_KEYS = tuple(_SESSION_ENV_MAP.values())


def _has_context_files(path: Path) -> bool:
    """Return True when the workspace carries prompt context files."""
    return any((path / name).exists() for name in _CONTEXT_FILES)


def _resolve_context_cwd() -> str:
    """Pick the directory used for prompt context discovery."""
    cwd = Path.cwd().resolve()
    hermes_home = _hermes_home.resolve()
    if _has_context_files(hermes_home):
        return str(hermes_home)
    return str(cwd)


def _load_workspace_config() -> Dict[str, Any]:
    """Load the active HERMES_HOME config file."""
    config_path = _hermes_home / "config.yaml"
    if not config_path.exists():
        return {}

    try:
        import yaml

        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except UnicodeDecodeError:
        try:
            import yaml

            with config_path.open(encoding="latin-1") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            return {}
    except Exception:
        return {}

    return config if isinstance(config, dict) else {}


def _load_prefill_messages() -> List[Dict[str, Any]]:
    """Load ephemeral prefill messages from env/config, mirroring gateway behavior."""
    file_path = str(os.getenv("HERMES_PREFILL_MESSAGES_FILE", "") or "").strip()
    if not file_path:
        file_path = str(_load_workspace_config().get("prefill_messages_file", "") or "").strip()
    if not file_path:
        return []

    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = _hermes_home / path
    if not path.exists():
        logger.warning("Prefill messages file not found: %s", path)
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to load prefill messages from %s: %s", path, e)
        return []

    if not isinstance(data, list):
        logger.warning("Prefill messages file must contain a JSON array: %s", path)
        return []
    return data


def _load_ephemeral_system_prompt() -> str:
    """Load the workspace/system prompt override for cron runs."""
    prompt = str(os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "") or "").strip()
    if prompt:
        return prompt
    agent_cfg = _load_workspace_config().get("agent", {})
    if not isinstance(agent_cfg, dict):
        return ""
    return str(agent_cfg.get("system_prompt", "") or "").strip()


def _load_reasoning_config() -> Optional[Dict[str, Any]]:
    """Load reasoning config from env/config, mirroring gateway behavior."""
    effort = str(os.getenv("HERMES_REASONING_EFFORT", "") or "").strip()
    if not effort:
        agent_cfg = _load_workspace_config().get("agent", {})
        if isinstance(agent_cfg, dict):
            effort = str(agent_cfg.get("reasoning_effort", "") or "").strip()
    if not effort:
        return None

    effort = effort.lower().strip()
    if effort == "none":
        return {"enabled": False}

    valid = ("xhigh", "high", "medium", "low", "minimal")
    if effort in valid:
        return {"enabled": True, "effort": effort}

    logger.warning("Unknown reasoning_effort '%s', using default (xhigh)", effort)
    return None


def _resolve_runtime_platform(job: dict) -> str:
    """Pick the platform profile cron should emulate for this job."""
    origin = _resolve_origin(job)
    platform = str((origin or {}).get("platform", "") or "").strip().lower()
    if not platform:
        deliver = str(job.get("deliver", "") or "").strip().lower()
        if deliver and deliver != "origin":
            platform = deliver.split(":", 1)[0].strip()

    if platform in ("", "origin", "local"):
        return "cli"
    return platform


def _load_enabled_toolsets(platform_key: str) -> Optional[List[str]]:
    """Resolve per-platform toolsets from config, with gateway-compatible defaults."""
    normalized = (platform_key or "cli").strip().lower()
    config_toolsets = _load_workspace_config().get("platform_toolsets", {})
    if isinstance(config_toolsets, dict):
        configured = config_toolsets.get(normalized)
        if isinstance(configured, list):
            resolved = [str(item).strip() for item in configured if str(item).strip()]
            if resolved:
                return resolved

    default_toolset_map = {
        "cli": "hermes-cli",
        "telegram": "hermes-telegram",
        "discord": "hermes-discord",
        "whatsapp": "hermes-whatsapp",
        "slack": "hermes-slack",
    }
    default_toolset = default_toolset_map.get(normalized)
    return [default_toolset] if default_toolset else None


def _clear_job_session_env() -> None:
    """Clear origin/session environment variables between cron runs."""
    for key in _SESSION_ENV_KEYS:
        os.environ.pop(key, None)


def _set_job_session_env(origin: Optional[dict]) -> None:
    """Set origin/session environment variables for tool availability checks."""
    _clear_job_session_env()
    if not origin:
        return

    for origin_key, env_key in _SESSION_ENV_MAP.items():
        value = origin.get(origin_key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            os.environ[env_key] = text


def _bridge_terminal_config_from_yaml() -> None:
    """Mirror terminal config.yaml settings into TERMINAL_* env vars."""
    config_path = _hermes_home / "config.yaml"
    if not config_path.exists():
        return

    try:
        import yaml

        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return

    terminal_cfg = config.get("terminal", {})
    if not isinstance(terminal_cfg, dict):
        return

    env_map = {
        "backend": "TERMINAL_ENV",
        "cwd": "TERMINAL_CWD",
        "timeout": "TERMINAL_TIMEOUT",
        "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
        "docker_image": "TERMINAL_DOCKER_IMAGE",
        "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
        "modal_image": "TERMINAL_MODAL_IMAGE",
        "ssh_host": "TERMINAL_SSH_HOST",
        "ssh_user": "TERMINAL_SSH_USER",
        "ssh_port": "TERMINAL_SSH_PORT",
        "ssh_key": "TERMINAL_SSH_KEY",
        "container_cpu": "TERMINAL_CONTAINER_CPU",
        "container_memory": "TERMINAL_CONTAINER_MEMORY",
        "container_disk": "TERMINAL_CONTAINER_DISK",
        "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
        "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
    }

    for cfg_key, env_var in env_map.items():
        if cfg_key not in terminal_cfg:
            continue
        value = terminal_cfg[cfg_key]
        if isinstance(value, list):
            os.environ[env_var] = json.dumps(value)
        else:
            os.environ[env_var] = str(value)


def _normalize_terminal_env() -> None:
    """Apply messaging terminal env defaults for cron runs."""
    _bridge_terminal_config_from_yaml()
    messaging_cwd = os.getenv("MESSAGING_CWD")
    if messaging_cwd:
        os.environ["TERMINAL_CWD"] = messaging_cwd
    elif not os.getenv("TERMINAL_CWD"):
        os.environ["TERMINAL_CWD"] = str(Path.home())


def _select_api_key(target_base_url: str, provider_hint: str = "") -> str:
    """Pick the credential that matches the effective provider endpoint."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    provider_hint = (provider_hint or "").strip().lower()
    target_base = (target_base_url or "").lower()

    if provider_hint == "openrouter" or "openrouter.ai" in target_base:
        return openrouter_key or openai_key
    return openai_key or openrouter_key


def _render_job_output(
    *,
    job_name: str,
    job_id: str,
    schedule_display: str,
    prompt: str,
    run_time: str,
    response: str = "",
    error: str = "",
    traceback_text: str = "",
) -> str:
    """Render a saved cron output document."""
    if error:
        error_block = error
        if traceback_text:
            error_block = f"{error}\n\n{traceback_text}"
        return f"""# Cron Job: {job_name} (FAILED)

**Job ID:** {job_id}
**Run Time:** {run_time}
**Schedule:** {schedule_display}

## Prompt

{prompt}

## Error

```
{error_block}
```
"""

    return f"""# Cron Job: {job_name}

**Job ID:** {job_id}
**Run Time:** {run_time}
**Schedule:** {schedule_display}

## Prompt

{prompt}

## Response

{response}
"""


def _resolve_origin(job: dict) -> Optional[dict]:
    """Extract origin info from a job, returning the stored origin dict or None."""
    origin = job.get("origin")
    if not origin:
        return None
    platform = origin.get("platform")
    chat_id = origin.get("chat_id")
    if platform and chat_id:
        return origin
    return None


def _deliver_result(job: dict, content: str) -> None:
    """
    Deliver job output to the configured target (origin chat, specific platform, etc.).

    Uses the standalone platform send functions from send_message_tool so delivery
    works whether or not the gateway is running.
    """
    deliver = job.get("deliver", "local")
    origin = _resolve_origin(job)

    if deliver == "local":
        return

    # Resolve target platform + chat_id
    if deliver == "origin":
        if not origin:
            logger.warning("Job '%s' deliver=origin but no origin stored, skipping delivery", job["id"])
            return
        platform_name = origin["platform"]
        chat_id = origin["chat_id"]
    elif ":" in deliver:
        platform_name, chat_id = deliver.split(":", 1)
    else:
        # Bare platform name like "telegram" — need to resolve to origin or home channel
        platform_name = deliver
        if origin and origin.get("platform") == platform_name:
            chat_id = origin["chat_id"]
        else:
            # Fall back to home channel
            chat_id = os.getenv(f"{platform_name.upper()}_HOME_CHANNEL", "")
            if not chat_id:
                logger.warning("Job '%s' deliver=%s but no chat_id or home channel. Set via: hermes config set %s_HOME_CHANNEL <channel_id>", job["id"], deliver, platform_name.upper())
                return

    from tools.send_message_tool import _send_to_platform
    from gateway.config import load_gateway_config, Platform

    platform_map = {
        "telegram": Platform.TELEGRAM,
        "discord": Platform.DISCORD,
        "slack": Platform.SLACK,
        "whatsapp": Platform.WHATSAPP,
    }
    platform = platform_map.get(platform_name.lower())
    if not platform:
        logger.warning("Job '%s': unknown platform '%s' for delivery", job["id"], platform_name)
        return

    try:
        config = load_gateway_config()
    except Exception as e:
        logger.error("Job '%s': failed to load gateway config for delivery: %s", job["id"], e)
        return

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        logger.warning("Job '%s': platform '%s' not configured/enabled", job["id"], platform_name)
        return

    # Run the async send in a fresh event loop (safe from any thread)
    try:
        result = asyncio.run(_send_to_platform(platform, pconfig, chat_id, content))
    except RuntimeError:
        # asyncio.run() fails if there's already a running loop in this thread;
        # spin up a new thread to avoid that.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _send_to_platform(platform, pconfig, chat_id, content))
            result = future.result(timeout=30)
    except Exception as e:
        logger.error("Job '%s': delivery to %s:%s failed: %s", job["id"], platform_name, chat_id, e)
        return

    if result and result.get("error"):
        logger.error("Job '%s': delivery error: %s", job["id"], result["error"])
    else:
        logger.info("Job '%s': delivered to %s:%s", job["id"], platform_name, chat_id)
        # Mirror the delivered content into the target's gateway session
        try:
            from gateway.mirror import mirror_to_session
            mirror_to_session(platform_name, chat_id, content, source_label="cron")
        except Exception:
            pass


def run_job(job: dict) -> tuple[bool, str, str, Optional[str]]:
    """
    Execute a single cron job.
    
    Returns:
        Tuple of (success, full_output_doc, final_response, error_message)
    """
    from run_agent import AIAgent
    
    job_id = job["id"]
    job_name = job["name"]
    prompt = job["prompt"]
    origin = _resolve_origin(job)
    platform_key = _resolve_runtime_platform(job)

    logger.info("Running job '%s' (ID: %s)", job_name, job_id)
    logger.info("Prompt: %s", prompt[:100])

    # Inject origin context so platform-gated tools resolve the same way they do
    # in live gateway sessions.
    _set_job_session_env(origin)

    try:
        # Re-read .env and config.yaml fresh every run so provider/key
        # changes take effect without a gateway restart.
        from dotenv import load_dotenv
        try:
            load_dotenv(str(_hermes_home / ".env"), override=True, encoding="utf-8")
        except UnicodeDecodeError:
            load_dotenv(str(_hermes_home / ".env"), override=True, encoding="latin-1")
        _normalize_terminal_env()

        model = os.getenv("HERMES_MODEL", "anthropic/claude-opus-4.6")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = _select_api_key(base_url)
        model_extra_body = None
        enabled_toolsets = _load_enabled_toolsets(platform_key)
        prefill_messages = _load_prefill_messages()
        ephemeral_system_prompt = _load_ephemeral_system_prompt()
        reasoning_config = _load_reasoning_config()

        try:
            resolved = load_model_runtime_config(
                _hermes_home / "config.yaml",
                default_model=model,
                default_base_url=base_url,
                logger=logger,
            )
            model = resolved.model
            base_url = resolved.base_url
            api_key = _select_api_key(base_url, resolved.provider)
            model_extra_body = resolved.extra_body
            if resolved.provider == "nous":
                try:
                    from hermes_cli.auth import resolve_nous_runtime_credentials
                    creds = resolve_nous_runtime_credentials(min_key_ttl_seconds=5 * 60)
                    api_key = creds.get("api_key", api_key)
                    base_url = creds.get("base_url", base_url)
                except Exception as nous_err:
                    logging.warning("Nous Portal credential resolution failed for cron: %s", nous_err)
        except Exception:
            pass

        agent = AIAgent(
            model=model,
            api_key=api_key,
            base_url=base_url,
            model_extra_body=model_extra_body,
            enabled_toolsets=enabled_toolsets,
            quiet_mode=True,
            ephemeral_system_prompt=ephemeral_system_prompt or None,
            prefill_messages=prefill_messages or None,
            reasoning_config=reasoning_config,
            session_id=f"cron_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            platform=platform_key,
            context_cwd=_resolve_context_cwd(),
        )
        
        result = agent.run_conversation(prompt)

        final_response = (result.get("final_response") or "").strip()
        agent_error = (result.get("error") or "").strip()
        run_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if result.get("failed") or agent_error or not final_response:
            error_msg = agent_error or "No response generated"
            logger.error("Job '%s' failed: %s", job_name, error_msg)
            output = _render_job_output(
                job_name=job_name,
                job_id=job_id,
                schedule_display=job.get("schedule_display", "N/A"),
                prompt=prompt,
                run_time=run_time,
                error=error_msg,
            )
            return False, output, "", error_msg

        output = _render_job_output(
            job_name=job_name,
            job_id=job_id,
            schedule_display=job.get("schedule_display", "N/A"),
            prompt=prompt,
            run_time=run_time,
            response=final_response,
        )

        logger.info("Job '%s' completed successfully", job_name)
        return True, output, final_response, None
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error("Job '%s' failed: %s", job_name, error_msg)

        output = _render_job_output(
            job_name=job_name,
            job_id=job_id,
            schedule_display=job.get("schedule_display", "N/A"),
            prompt=prompt,
            run_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            error=error_msg,
            traceback_text=traceback.format_exc(),
        )
        return False, output, "", error_msg

    finally:
        # Clean up injected env vars so they don't leak to other jobs
        _clear_job_session_env()


def tick(verbose: bool = True) -> int:
    """
    Check and run all due jobs.
    
    Uses a file lock so only one tick runs at a time, even if the gateway's
    in-process ticker and a standalone daemon or manual tick overlap.
    
    Args:
        verbose: Whether to print status messages
    
    Returns:
        Number of jobs executed (0 if another tick is already running)
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)

    # Cross-platform file locking: fcntl on Unix, msvcrt on Windows
    try:
        lock_fd = open(_LOCK_FILE, "w")
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt:
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
    except (OSError, IOError):
        logger.debug("Tick skipped — another instance holds the lock")
        return 0

    try:
        due_jobs = get_due_jobs()

        if verbose and not due_jobs:
            logger.info("%s - No jobs due", datetime.now().strftime('%H:%M:%S'))
            return 0

        if verbose:
            logger.info("%s - %s job(s) due", datetime.now().strftime('%H:%M:%S'), len(due_jobs))

        executed = 0
        for job in due_jobs:
            try:
                success, output, final_response, error = run_job(job)

                output_file = save_job_output(job["id"], output)
                if verbose:
                    logger.info("Output saved to: %s", output_file)

                # Deliver the final response to the origin/target chat
                deliver_content = final_response if success else f"⚠️ Cron job '{job.get('name', job['id'])}' failed:\n{error}"
                if deliver_content:
                    try:
                        _deliver_result(job, deliver_content)
                    except Exception as de:
                        logger.error("Delivery failed for job %s: %s", job["id"], de)

                mark_job_run(job["id"], success, error)
                executed += 1

            except Exception as e:
                logger.error("Error processing job %s: %s", job['id'], e)
                mark_job_run(job["id"], False, str(e))

        return executed
    finally:
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        elif msvcrt:
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
        lock_fd.close()


if __name__ == "__main__":
    tick(verbose=True)
