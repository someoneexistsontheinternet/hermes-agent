"""
Gateway runner - entry point for messaging platform integrations.

This module provides:
- start_gateway(): Start all configured platform adapters
- GatewayRunner: Main class managing the gateway lifecycle

Usage:
    # Start the gateway
    python -m gateway.run
    
    # Or from CLI
    python cli.py --gateway
"""

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import sys
import signal
import threading
import urllib.parse
import urllib.request
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Resolve Hermes home directory (respects HERMES_HOME override)
_hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))

# Load environment variables from ~/.hermes/.env first
from dotenv import load_dotenv
_env_path = _hermes_home / '.env'
if _env_path.exists():
    try:
        # Override inherited service env vars (e.g. launchd/systemd) so
        # ~/.hermes/.env is the source of truth for gateway auth flags.
        load_dotenv(_env_path, override=True, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, override=True, encoding="latin-1")
# Also try project .env as fallback
load_dotenv()

# Bridge config.yaml values into the environment so os.getenv() picks them up.
# config.yaml is authoritative for terminal settings — overrides .env.
_config_path = _hermes_home / 'config.yaml'
if _config_path.exists():
    try:
        import yaml as _yaml
        with open(_config_path) as _f:
            _cfg = _yaml.safe_load(_f) or {}
        # Top-level simple values (fallback only — don't override .env)
        for _key, _val in _cfg.items():
            if isinstance(_val, (str, int, float, bool)) and _key not in os.environ:
                os.environ[_key] = str(_val)
        # Terminal config is nested — bridge to TERMINAL_* env vars.
        # config.yaml overrides .env for these since it's the documented config path.
        _terminal_cfg = _cfg.get("terminal", {})
        if _terminal_cfg and isinstance(_terminal_cfg, dict):
            _terminal_env_map = {
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
            for _cfg_key, _env_var in _terminal_env_map.items():
                if _cfg_key in _terminal_cfg:
                    _val = _terminal_cfg[_cfg_key]
                    if isinstance(_val, list):
                        os.environ[_env_var] = json.dumps(_val)
                    else:
                        os.environ[_env_var] = str(_val)
    except Exception:
        pass  # Non-fatal; gateway can still run with .env values

# Gateway runs in quiet mode - suppress debug output and use cwd directly (no temp dirs)
os.environ["HERMES_QUIET"] = "1"

# Enable interactive exec approval for dangerous commands on messaging platforms
os.environ["HERMES_EXEC_ASK"] = "1"

# Set terminal working directory for messaging platforms
# Uses MESSAGING_CWD if set, otherwise defaults to home directory
# This is separate from CLI which uses the directory where `hermes` is run
messaging_cwd = os.getenv("MESSAGING_CWD") or str(Path.home())
os.environ["TERMINAL_CWD"] = messaging_cwd

from gateway.config import (
    Platform,
    GatewayConfig,
    load_gateway_config,
)
from gateway.session import (
    SessionStore,
    SessionEntry,
    SessionSource,
    SessionContext,
    build_session_context,
    build_session_context_prompt,
)
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from model_runtime_config import load_model_runtime_config

logger = logging.getLogger(__name__)


def _has_context_files(path: Path) -> bool:
    """Return True when a directory contains prompt context files."""
    return any(
        [
            (path / "AGENTS.md").exists(),
            (path / "agents.md").exists(),
            (path / "SOUL.md").exists(),
            (path / "soul.md").exists(),
            (path / ".cursorrules").exists(),
            (path / ".cursor" / "rules").is_dir(),
        ]
    )


class GatewayRunner:
    """
    Main gateway controller.
    
    Manages the lifecycle of all platform adapters and routes
    messages to/from the agent.
    """
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or load_gateway_config()
        self.adapters: Dict[Platform, BasePlatformAdapter] = {}
        self._context_cwd = self._resolve_context_cwd()

        # Load ephemeral config from config.yaml / env vars.
        # Both are injected at API-call time only and never persisted.
        self._prefill_messages = self._load_prefill_messages()
        self._ephemeral_system_prompt = self._load_ephemeral_system_prompt()
        self._reasoning_config = self._load_reasoning_config()

        # Wire process registry into session store for reset protection
        from tools.process_registry import process_registry
        self.session_store = SessionStore(
            self.config.sessions_dir, self.config,
            has_active_processes_fn=lambda key: process_registry.has_active_for_session(key),
            on_auto_reset=self._flush_memories_before_reset,
        )
        self.delivery_router = DeliveryRouter(self.config)
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Track running agents per session for interrupt support
        # Key: session_key, Value: AIAgent instance
        self._running_agents: Dict[str, Any] = {}
        self._pending_messages: Dict[str, str] = {}  # Queued messages during interrupt
        
        # Track pending exec approvals per session
        # Key: session_key, Value: {"command": str, "pattern_key": str}
        self._pending_approvals: Dict[str, Dict[str, str]] = {}
        
        # Initialize session database for session_search tool support
        self._session_db = None
        try:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        except Exception as e:
            logger.debug("SQLite session store not available: %s", e)
        
        # DM pairing store for code-based user authorization
        from gateway.pairing import PairingStore
        self.pairing_store = PairingStore()
        
        # Event hook system
        from gateway.hooks import HookRegistry
        self.hooks = HookRegistry()

    def _resolve_context_cwd(self) -> str:
        """Pick the host directory used for gateway prompt context files."""
        cwd = Path.cwd().resolve()
        hermes_home = _hermes_home.resolve()

        # Gateway is often launched from the install repo or a service manager.
        # If the workspace under HERMES_HOME carries AGENTS/SOUL/Cursor rules,
        # prefer that over the process cwd so prompt context matches the gateway
        # workspace instead of the launch directory.
        if _has_context_files(hermes_home):
            return str(hermes_home)
        return str(cwd)

    def _connected_platforms(self) -> List[Platform]:
        """Return platforms whose adapters currently report a live connection."""
        connected: List[Platform] = []
        for platform, adapter in self.adapters.items():
            try:
                if bool(getattr(adapter, "is_connected", True)):
                    connected.append(platform)
            except Exception:
                continue
        return connected

    def _live_discord_adapter(self) -> Optional[BasePlatformAdapter]:
        """Return the Discord adapter when it is live and connected."""
        adapter = self.adapters.get(Platform.DISCORD)
        if adapter is None:
            return None
        try:
            if not bool(getattr(adapter, "is_connected", True)):
                return None
        except Exception:
            return None
        return adapter

    def _discord_fork_available(self, source: SessionSource) -> bool:
        """Return True when this source can use live Discord thread forking."""
        if source.platform != Platform.DISCORD or source.chat_type not in {"group", "channel"}:
            return False

        adapter = self._live_discord_adapter()
        if adapter is None:
            return False

        checker = getattr(adapter, "is_auto_fork_available", None)
        return callable(checker) and checker(source)
    
    def _flush_memories_before_reset(self, old_entry):
        """Prompt the agent to save memories/skills before an auto-reset.
        
        Called synchronously by SessionStore before destroying an expired session.
        Loads the transcript, gives the agent a real turn with memory + skills
        tools, and explicitly asks it to preserve anything worth keeping.
        """
        try:
            history = self.session_store.load_transcript(old_entry.session_id)
            if not history or len(history) < 4:
                return

            from run_agent import AIAgent
            _flush_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
            _flush_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            _flush_model = os.getenv("HERMES_MODEL") or os.getenv("LLM_MODEL", "anthropic/claude-opus-4.6")

            if not _flush_api_key:
                return

            tmp_agent = AIAgent(
                model=_flush_model,
                api_key=_flush_api_key,
                base_url=_flush_base_url,
                max_iterations=8,
                quiet_mode=True,
                enabled_toolsets=["memory", "skills"],
                session_id=old_entry.session_id,
                context_cwd=self._context_cwd,
            )

            # Build conversation history from transcript
            msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]

            # Give the agent a real turn to think about what to save
            flush_prompt = (
                "[System: This session is about to be automatically reset due to "
                "inactivity or a scheduled daily reset. The conversation context "
                "will be cleared after this turn.\n\n"
                "Review the conversation above and:\n"
                "1. Save any important facts, preferences, or decisions to memory "
                "(user profile or your notes) that would be useful in future sessions.\n"
                "2. If you discovered a reusable workflow or solved a non-trivial "
                "problem, consider saving it as a skill.\n"
                "3. If nothing is worth saving, that's fine — just skip.\n\n"
                "Do NOT respond to the user. Just use the memory and skill_manage "
                "tools if needed, then stop.]"
            )

            tmp_agent.run_conversation(
                user_message=flush_prompt,
                conversation_history=msgs,
            )
            logger.info("Pre-reset save completed for session %s", old_entry.session_id)
        except Exception as e:
            logger.debug("Pre-reset save failed for session %s: %s", old_entry.session_id, e)

    @staticmethod
    def _sandbox_task_id_for_session(
        session_key: Optional[str],
        source: SessionSource,
        session_id: str,
    ) -> str:
        """Build a stable, filesystem-safe task_id for persistent sandboxes.

        Keying by session_key keeps one sandbox per chat/session even when
        session_id rotates on /new or automatic resets.
        """
        stable_key = session_key or (
            f"{source.platform.value}:{source.chat_type}:{source.chat_id}:"
            f"{source.thread_id or ''}:{source.user_id or ''}:{session_id}"
        )
        digest = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:24]
        platform = source.platform.value if source and source.platform else "unknown"
        return f"gw-{platform}-{digest}"
    
    @staticmethod
    def _load_prefill_messages() -> List[Dict[str, Any]]:
        """Load ephemeral prefill messages from config or env var.
        
        Checks HERMES_PREFILL_MESSAGES_FILE env var first, then falls back to
        the prefill_messages_file key in ~/.hermes/config.yaml.
        Relative paths are resolved from ~/.hermes/.
        """
        import json as _json
        file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
        if not file_path:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path) as _f:
                        cfg = _y.safe_load(_f) or {}
                    file_path = cfg.get("prefill_messages_file", "")
            except Exception:
                pass
        if not file_path:
            return []
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = _hermes_home / path
        if not path.exists():
            logger.warning("Prefill messages file not found: %s", path)
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            if not isinstance(data, list):
                logger.warning("Prefill messages file must contain a JSON array: %s", path)
                return []
            return data
        except Exception as e:
            logger.warning("Failed to load prefill messages from %s: %s", path, e)
            return []

    @staticmethod
    def _load_ephemeral_system_prompt() -> str:
        """Load ephemeral system prompt from config or env var.
        
        Checks HERMES_EPHEMERAL_SYSTEM_PROMPT env var first, then falls back to
        agent.system_prompt in ~/.hermes/config.yaml.
        """
        prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
        if prompt:
            return prompt
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path) as _f:
                    cfg = _y.safe_load(_f) or {}
                return (cfg.get("agent", {}).get("system_prompt", "") or "").strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _load_reasoning_config() -> dict | None:
        """Load reasoning effort from config or env var.
        
        Checks HERMES_REASONING_EFFORT env var first, then agent.reasoning_effort
        in config.yaml. Valid: "xhigh", "high", "medium", "low", "minimal", "none".
        Returns None to use default (xhigh).
        """
        effort = os.getenv("HERMES_REASONING_EFFORT", "")
        if not effort:
            try:
                import yaml as _y
                cfg_path = _hermes_home / "config.yaml"
                if cfg_path.exists():
                    with open(cfg_path) as _f:
                        cfg = _y.safe_load(_f) or {}
                    effort = str(cfg.get("agent", {}).get("reasoning_effort", "") or "").strip()
            except Exception:
                pass
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

    async def start(self) -> bool:
        """
        Start the gateway and all configured platform adapters.
        
        Returns True if at least one adapter connected successfully.
        """
        logger.info("Starting Hermes Gateway...")
        logger.info("Session storage: %s", self.config.sessions_dir)
        
        # Warn if no user allowlists are configured and open access is not opted in
        _any_allowlist = any(
            os.getenv(v)
            for v in ("TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
                       "WHATSAPP_ALLOWED_USERS", "SLACK_ALLOWED_USERS",
                       "GATEWAY_ALLOWED_USERS")
        )
        _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes")
        if not _any_allowlist and not _allow_all:
            logger.warning(
                "No user allowlists configured. All unauthorized users will be denied. "
                "Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access, "
                "or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id)."
            )
        
        # Discover and load event hooks
        self.hooks.discover_and_load()
        
        # Recover background processes from checkpoint (crash recovery)
        try:
            from tools.process_registry import process_registry
            recovered = process_registry.recover_from_checkpoint()
            if recovered:
                logger.info("Recovered %s background process(es) from previous run", recovered)
        except Exception as e:
            logger.warning("Process checkpoint recovery: %s", e)
        
        connected_count = 0
        
        # Initialize and connect each configured platform
        for platform, platform_config in self.config.platforms.items():
            if not platform_config.enabled:
                continue
            
            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                logger.warning("No adapter available for %s", platform.value)
                continue
            
            # Set up message handler
            adapter.set_message_handler(self._handle_message)
            
            # Try to connect
            logger.info("Connecting to %s...", platform.value)
            try:
                success = await adapter.connect()
                if success:
                    self.adapters[platform] = adapter
                    connected_count += 1
                    logger.info("✓ %s connected", platform.value)
                else:
                    logger.warning("✗ %s failed to connect", platform.value)
            except Exception as e:
                logger.error("✗ %s error: %s", platform.value, e)
        
        if connected_count == 0:
            logger.warning("No messaging platforms connected.")
            logger.info("Gateway will continue running for cron job execution.")
        
        # Update delivery router with adapters
        self.delivery_router.adapters = self.adapters
        
        self._running = True
        
        # Emit gateway:startup hook
        hook_count = len(self.hooks.loaded_hooks)
        if hook_count:
            logger.info("%s hook(s) loaded", hook_count)
        await self.hooks.emit("gateway:startup", {
            "platforms": [p.value for p in self.adapters.keys()],
        })
        
        if connected_count > 0:
            logger.info("Gateway running with %s platform(s)", connected_count)
        
        # Build initial channel directory for send_message name resolution
        try:
            from gateway.channel_directory import build_channel_directory
            directory = build_channel_directory(self.adapters)
            ch_count = sum(len(chs) for chs in directory.get("platforms", {}).values())
            logger.info("Channel directory built: %d target(s)", ch_count)
        except Exception as e:
            logger.warning("Channel directory build failed: %s", e)
        
        logger.info("Press Ctrl+C to stop")
        
        return True
    
    async def stop(self) -> None:
        """Stop the gateway and disconnect all adapters."""
        logger.info("Stopping gateway...")
        self._running = False
        
        for platform, adapter in self.adapters.items():
            try:
                await adapter.disconnect()
                logger.info("✓ %s disconnected", platform.value)
            except Exception as e:
                logger.error("✗ %s disconnect error: %s", platform.value, e)
        
        self.adapters.clear()
        self._shutdown_event.set()
        
        from gateway.status import remove_pid_file
        remove_pid_file()
        
        logger.info("Gateway stopped")
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()
    
    def _create_adapter(
        self, 
        platform: Platform, 
        config: Any
    ) -> Optional[BasePlatformAdapter]:
        """Create the appropriate adapter for a platform."""
        if platform == Platform.TELEGRAM:
            from gateway.platforms.telegram import TelegramAdapter, check_telegram_requirements
            if not check_telegram_requirements():
                logger.warning("Telegram: python-telegram-bot not installed")
                return None
            return TelegramAdapter(config)
        
        elif platform == Platform.DISCORD:
            from gateway.platforms.discord import DiscordAdapter, check_discord_requirements
            if not check_discord_requirements():
                logger.warning("Discord: discord.py not installed")
                return None
            return DiscordAdapter(config)
        
        elif platform == Platform.WHATSAPP:
            from gateway.platforms.whatsapp import WhatsAppAdapter, check_whatsapp_requirements
            if not check_whatsapp_requirements():
                logger.warning("WhatsApp: Node.js not installed or bridge not configured")
                return None
            return WhatsAppAdapter(config)
        
        elif platform == Platform.SLACK:
            from gateway.platforms.slack import SlackAdapter, check_slack_requirements
            if not check_slack_requirements():
                logger.warning("Slack: slack-bolt not installed. Run: pip install 'hermes-agent[slack]'")
                return None
            return SlackAdapter(config)
        
        return None

    @staticmethod
    def _is_truthy(value: Any) -> bool:
        """Parse flexible truthy values from env/config strings."""
        if value is None:
            return False
        return str(value).strip().strip('"').strip("'").lower() in (
            "true", "1", "yes", "y", "on"
        )

    def _discord_dms_enabled(self) -> bool:
        """Return whether Discord DMs are enabled for this gateway process."""
        env_val = (os.getenv("DISCORD_ENABLE_DMS", "") or "").strip()
        if env_val:
            return self._is_truthy(env_val)

        pconfig = self.config.platforms.get(Platform.DISCORD)
        if not pconfig or not isinstance(pconfig.extra, dict):
            return True
        if "enable_dms" in pconfig.extra:
            return self._is_truthy(pconfig.extra.get("enable_dms"))
        return True

    def _discord_cost_summary_enabled(self) -> bool:
        """Return whether Discord responses should include a request cost footer."""
        env_val = (os.getenv("DISCORD_COST_SUMMARY_ENABLED", "") or "").strip()
        if env_val:
            return self._is_truthy(env_val)

        pconfig = self.config.platforms.get(Platform.DISCORD)
        if not pconfig or not isinstance(pconfig.extra, dict):
            return True
        if "cost_summary_enabled" in pconfig.extra:
            return self._is_truthy(pconfig.extra.get("cost_summary_enabled"))
        return True

    @staticmethod
    def _format_cost_usd(cost_usd: float) -> str:
        """Format USD cost compactly without hiding sub-cent request costs."""
        abs_cost = abs(cost_usd)
        if abs_cost >= 0.01:
            return f"${cost_usd:.2f}"
        if abs_cost >= 0.001:
            return f"${cost_usd:.3f}"
        return f"${cost_usd:.4f}"

    def _append_request_cost_summary(
        self,
        response: str,
        agent_result: Dict[str, Any],
        source: SessionSource,
    ) -> str:
        """Append a Discord-only request cost footer when usage data includes USD cost."""
        if not response or source.platform != Platform.DISCORD:
            return response
        if not self._discord_cost_summary_enabled():
            return response

        usage = agent_result.get("request_usage") or {}
        if not isinstance(usage, dict):
            return response

        raw_cost = usage.get("cost_usd")
        if raw_cost is None:
            return response

        try:
            cost_usd = float(raw_cost)
        except (TypeError, ValueError):
            return response

        summary = f"-# {self._format_cost_usd(cost_usd)} USD spent"
        trimmed = response.rstrip()
        if trimmed.endswith(summary):
            return response
        if not trimmed:
            return summary
        return f"{trimmed}\n\n{summary}"

    @staticmethod
    def _new_delivery_state(
        source: SessionSource,
        event: Optional[MessageEvent] = None,
    ) -> Dict[str, Any]:
        """Create a fresh delivery-state object for one inbound event."""
        return {
            "chat_id": source.chat_id,
            "reply_to": getattr(event, "message_id", None),
            "thread_result": None,
            "thread_source": None,
            "thread_session_id": None,
            "thread_session_key": None,
            "transcript_notice": None,
            "main_notice": None,
            "main_notice_sent": False,
            "thread_transcript_recorded": False,
        }

    @staticmethod
    def _same_speaker(left: Optional[SessionSource], right: Optional[SessionSource]) -> bool:
        """Return True when two sources appear to be the same human speaker."""
        if left is None or right is None:
            return False

        left_id = str(getattr(left, "user_id", "") or "").strip()
        right_id = str(getattr(right, "user_id", "") or "").strip()
        if left_id and right_id:
            return left_id == right_id

        left_name = str(getattr(left, "user_name", "") or "").strip()
        right_name = str(getattr(right, "user_name", "") or "").strip()
        return bool(left_name and right_name and left_name == right_name)

    @staticmethod
    def _discord_interrupt_note(
        previous_source: Optional[SessionSource],
        current_source: Optional[SessionSource],
    ) -> str:
        """Build a short system note when another Discord user takes over a turn."""
        if current_source is None or current_source.platform != Platform.DISCORD:
            return ""
        current_user = str(current_source.user_name or current_source.user_id or "unknown").strip() or "unknown"
        previous_user = str(
            getattr(previous_source, "user_name", None)
            or getattr(previous_source, "user_id", None)
            or "another user"
        ).strip() or "another user"
        return (
            "[System note: The previous in-progress reply for this Discord chat "
            f"(started for {previous_user}) was interrupted by a new message from {current_user}. "
            "Ignore the interrupted task and answer the new speaker instead.]"
        )

    def _discord_reply_context_from_event(
        self,
        event: Optional[MessageEvent],
        source: Optional[SessionSource],
    ) -> Optional[Dict[str, Any]]:
        """Resolve author + preview for the message this Discord turn replies to."""
        if event is None or source is None or source.platform != Platform.DISCORD:
            return None

        reply_id = str(getattr(event, "reply_to_message_id", "") or "").strip()
        if not reply_id:
            return None

        adapter = self._live_discord_adapter()
        archive_db = getattr(adapter, "_archive_db", None) if adapter else None
        resolver = getattr(archive_db, "get_reply_context", None)
        if not callable(resolver):
            return None

        reference = getattr(getattr(event, "raw_message", None), "reference", None)
        reply_channel_id = str(getattr(reference, "channel_id", "") or "").strip()
        if not reply_channel_id:
            reply_channel_id = str(source.chat_id or "").strip()

        try:
            return resolver(message_id=reply_id, channel_id=reply_channel_id or None)
        except Exception as e:
            logger.debug("Discord reply-context lookup failed (%s): %s", reply_id, e)
            return None

    async def _build_event_message_payload(
        self,
        event: MessageEvent,
        source: SessionSource,
        *,
        interrupt_note: str = "",
    ) -> tuple[Any, str]:
        """Build the user payload sent to the model for one inbound event."""
        message_text = event.text or ""
        image_items = self._collect_multimodal_image_items(event)
        supports_direct_multimodal = False
        if image_items:
            model_name = self._current_model_name()
            supports_direct_multimodal = self._model_supports_direct_multimodal(model_name)
            if not supports_direct_multimodal:
                image_paths = [item["path"] for item in image_items]
                message_text = await self._enrich_message_with_vision(
                    message_text, image_paths
                )

        if event.media_urls:
            audio_paths = []
            for i, path in enumerate(event.media_urls):
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                is_audio = (
                    mtype.startswith("audio/")
                    or event.message_type in (MessageType.VOICE, MessageType.AUDIO)
                )
                if is_audio:
                    audio_paths.append(path)
            if audio_paths:
                message_text = await self._enrich_message_with_transcription(
                    message_text, audio_paths
                )

        if event.media_urls and event.message_type == MessageType.DOCUMENT:
            for i, path in enumerate(event.media_urls):
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                if not (mtype.startswith("application/") or mtype.startswith("text/")):
                    continue
                import os as _os
                basename = _os.path.basename(path)
                parts = basename.split("_", 2)
                display_name = parts[2] if len(parts) >= 3 else basename
                import re as _re
                display_name = _re.sub(r"[^\w.\- ]", "_", display_name)

                if mtype.startswith("text/"):
                    context_note = (
                        f"[The user sent a text document: '{display_name}'. "
                        f"Its content has been included below. "
                        f"The file is also saved at: {path}]"
                    )
                else:
                    context_note = (
                        f"[The user sent a document: '{display_name}'. "
                        f"The file is saved at: {path}. "
                        f"Ask the user what they'd like you to do with it.]"
                    )
                message_text = f"{context_note}\n\n{message_text}"

        extra_context = (getattr(event, "extra_context", "") or "").strip()
        prefix_parts: List[str] = []
        if interrupt_note:
            prefix_parts.append(interrupt_note.strip())

        if extra_context:
            use_discord_context_only = (
                source.platform == Platform.DISCORD
                and event.message_type == MessageType.TEXT
                and not event.media_urls
            )
            if use_discord_context_only:
                reply_context = self._discord_reply_context_from_event(event, source)
                request_summary = self._discord_request_summary(
                    source,
                    event.text or "",
                    reply_context=reply_context,
                )
                prefix_parts.extend([extra_context, request_summary])
                message_text = "\n\n".join(part for part in prefix_parts if part).strip()
            else:
                prefix_parts.extend([extra_context, message_text])
                message_text = "\n\n".join(part for part in prefix_parts if part).strip()
        elif prefix_parts:
            prefix_parts.append(message_text)
            message_text = "\n\n".join(part for part in prefix_parts if part).strip()

        message_payload: Any = message_text
        if image_items and supports_direct_multimodal:
            multimodal_payload = self._build_multimodal_user_payload(message_text, image_items)
            if multimodal_payload:
                message_payload = multimodal_payload
            else:
                image_paths = [item["path"] for item in image_items]
                if image_paths:
                    message_text = await self._enrich_message_with_vision(
                        message_text, image_paths
                    )
                    message_payload = message_text

        return message_payload, message_text

    async def _prepare_agent_turn_state(
        self,
        source: SessionSource,
        event: MessageEvent,
    ) -> Dict[str, Any]:
        """Resolve session, prompt, history, and user payload for one inbound turn."""
        force_auto_reset = bool(getattr(event, "force_auto_reset", False))
        auto_reset_reason = str(getattr(event, "auto_reset_reason", "") or "").strip()
        session_entry = self.session_store.get_or_create_session(
            source,
            force_auto_reset=force_auto_reset,
        )
        session_key = session_entry.session_key

        context = build_session_context(
            source,
            self.config,
            session_entry,
            connected_platforms=self._connected_platforms(),
        )
        self._set_session_env(context)

        context_prompt = build_session_context_prompt(context)
        auto_fork_note = self._discord_auto_fork_prompt_note(source)
        if auto_fork_note:
            context_prompt = f"{context_prompt}\n\n{auto_fork_note}"

        if getattr(session_entry, "was_auto_reset", False):
            if auto_reset_reason:
                note = (
                    "[System note: The previous session was automatically reset "
                    f"before this turn ({auto_reset_reason}). "
                    "This is a fresh conversation with no prior context.]"
                )
            else:
                note = (
                    "[System note: The user's previous session expired due to inactivity. "
                    "This is a fresh conversation with no prior context.]"
                )
            context_prompt = note + "\n\n" + context_prompt
            session_entry.was_auto_reset = False

        history = self._collapse_trailing_fork_handoffs(
            self.session_store.load_transcript(session_entry.session_id),
            source=source,
        )
        message_payload, message_text = await self._build_event_message_payload(
            event,
            source,
        )

        return {
            "session_entry": session_entry,
            "session_id": session_entry.session_id,
            "session_key": session_key,
            "context_prompt": context_prompt,
            "history": history,
            "message_payload": message_payload,
            "message_text": message_text,
        }
    
    def _is_user_authorized(self, source: SessionSource) -> bool:
        """
        Check if a user is authorized to use the bot.
        
        Checks in order:
        1. Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
        2. Environment variable allowlists (TELEGRAM_ALLOWED_USERS, etc.)
        3. DM pairing approved list
        4. Global allow-all (GATEWAY_ALLOW_ALL_USERS=true)
        5. Default: deny
        """
        user_id = source.user_id
        if not user_id:
            return False

        platform_env_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOWED_USERS",
            Platform.DISCORD: "DISCORD_ALLOWED_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOWED_USERS",
            Platform.SLACK: "SLACK_ALLOWED_USERS",
        }
        platform_allow_all_map = {
            Platform.TELEGRAM: "TELEGRAM_ALLOW_ALL_USERS",
            Platform.DISCORD: "DISCORD_ALLOW_ALL_USERS",
            Platform.WHATSAPP: "WHATSAPP_ALLOW_ALL_USERS",
            Platform.SLACK: "SLACK_ALLOW_ALL_USERS",
        }

        # Per-platform allow-all flag (e.g., DISCORD_ALLOW_ALL_USERS=true)
        platform_allow_all_var = platform_allow_all_map.get(source.platform, "")
        platform_allow_all_val = os.getenv(platform_allow_all_var, "") if platform_allow_all_var else ""
        if platform_allow_all_var and self._is_truthy(platform_allow_all_val):
            return True

        # Check pairing store (always checked, regardless of allowlists)
        platform_name = source.platform.value if source.platform else ""
        if self.pairing_store.is_approved(platform_name, user_id):
            return True

        # Check platform-specific and global allowlists
        platform_allowlist = os.getenv(platform_env_map.get(source.platform, ""), "").strip()
        global_allowlist = os.getenv("GATEWAY_ALLOWED_USERS", "").strip()

        if not platform_allowlist and not global_allowlist:
            # No allowlists configured -- check global allow-all flag
            gateway_allow_all_val = os.getenv("GATEWAY_ALLOW_ALL_USERS", "")
            allowed = self._is_truthy(gateway_allow_all_val)
            if not allowed:
                logger.debug(
                    "Auth deny (no allowlists, allow-all disabled): platform=%s user_id=%s "
                    "platform_allow_all_var=%s platform_allow_all_val=%r gateway_allow_all_val=%r",
                    source.platform.value if source.platform else "unknown",
                    user_id,
                    platform_allow_all_var,
                    platform_allow_all_val,
                    gateway_allow_all_val,
                )
            return allowed

        # Check if user is in any allowlist
        allowed_ids = set()
        if platform_allowlist:
            allowed_ids.update(uid.strip() for uid in platform_allowlist.split(",") if uid.strip())
        if global_allowlist:
            allowed_ids.update(uid.strip() for uid in global_allowlist.split(",") if uid.strip())

        # WhatsApp JIDs have @s.whatsapp.net suffix — strip it for comparison
        check_ids = {user_id}
        if "@" in user_id:
            check_ids.add(user_id.split("@")[0])
        allowed = bool(check_ids & allowed_ids)
        if not allowed:
            logger.debug(
                "Auth deny (allowlist mismatch): platform=%s user_id=%s check_ids=%s "
                "platform_allowlist=%r global_allowlist=%r",
                source.platform.value if source.platform else "unknown",
                user_id,
                sorted(check_ids),
                platform_allowlist,
                global_allowlist,
            )
        return allowed
    
    async def _handle_message(self, event: MessageEvent) -> Optional[str]:
        """
        Handle an incoming message from any platform.
        
        This is the core message processing pipeline:
        1. Check user authorization
        2. Check for commands (/new, /reset, etc.)
        3. Check for running agent and interrupt if needed
        4. Get or create session
        5. Build context for agent
        6. Run agent conversation
        7. Return response
        """
        source = event.source

        # Optional Discord DM hard-block (defense-in-depth in gateway layer).
        if (
            source.platform == Platform.DISCORD
            and source.chat_type == "dm"
            and not self._discord_dms_enabled()
        ):
            logger.info("Ignoring Discord DM because DISCORD_ENABLE_DMS is disabled")
            return None
        
        # Check if user is authorized
        if not self._is_user_authorized(source):
            p = source.platform.value if source.platform else "unknown"
            logger.warning(
                "Unauthorized user: %s (%s) on %s | DISCORD_ALLOW_ALL_USERS=%r "
                "GATEWAY_ALLOW_ALL_USERS=%r DISCORD_ALLOWED_USERS=%r GATEWAY_ALLOWED_USERS=%r",
                source.user_id,
                source.user_name,
                p,
                os.getenv("DISCORD_ALLOW_ALL_USERS"),
                os.getenv("GATEWAY_ALLOW_ALL_USERS"),
                os.getenv("DISCORD_ALLOWED_USERS"),
                os.getenv("GATEWAY_ALLOWED_USERS"),
            )
            # In DMs: offer pairing code unless DMs are disabled. In groups: silently ignore.
            if source.chat_type == "dm" and not (
                source.platform == Platform.DISCORD and not self._discord_dms_enabled()
            ):
                platform_name = source.platform.value if source.platform else "unknown"
                code = self.pairing_store.generate_code(
                    platform_name, source.user_id, source.user_name or ""
                )
                if code:
                    adapter = self.adapters.get(source.platform)
                    if adapter:
                        await adapter.send(
                            source.chat_id,
                            f"Hi~ I don't recognize you yet!\n\n"
                            f"Here's your pairing code: `{code}`\n\n"
                            f"Ask the bot owner to run:\n"
                            f"`hermes pairing approve {platform_name} {code}`"
                        )
                else:
                    adapter = self.adapters.get(source.platform)
                    if adapter:
                        await adapter.send(
                            source.chat_id,
                            "Too many pairing requests right now~ "
                            "Please try again later!"
                        )
            return None
        
        # PRIORITY: If an agent is already running for this session, interrupt it
        # immediately. This is before command parsing to minimize latency -- the
        # user's "stop" message reaches the agent as fast as possible.
        _quick_key = (
            f"agent:main:{source.platform.value}:{source.chat_type}:{source.chat_id}"
            if source.chat_type != "dm"
            else f"agent:main:{source.platform.value}:dm"
        )
        if _quick_key in self._running_agents:
            running_agent = self._running_agents[_quick_key]
            logger.debug("PRIORITY interrupt for session %s", _quick_key[:20])
            running_agent.interrupt(event.text)
            if _quick_key in self._pending_messages:
                self._pending_messages[_quick_key] += "\n" + event.text
            else:
                self._pending_messages[_quick_key] = event.text
            return None
        
        # Check for commands
        command = event.get_command()
        if command in ["new", "reset"]:
            return await self._handle_reset_command(event)
        
        if command == "help":
            return await self._handle_help_command(event)
        
        if command == "status":
            return await self._handle_status_command(event)
        
        if command == "stop":
            return await self._handle_stop_command(event)
        
        if command == "model":
            return await self._handle_model_command(event)
        
        if command == "personality":
            return await self._handle_personality_command(event)
        
        if command == "retry":
            return await self._handle_retry_command(event)
        
        if command == "undo":
            return await self._handle_undo_command(event)

        if command == "fork":
            return await self._handle_fork_command(event)
        
        if command in ["sethome", "set-home"]:
            return await self._handle_set_home_command(event)
        
        # Check for pending exec approval responses
        if source.chat_type != "dm":
            session_key_preview = f"agent:main:{source.platform.value}:{source.chat_type}:{source.chat_id}"
        elif source.platform and source.platform.value == "whatsapp" and source.chat_id:
            session_key_preview = f"agent:main:{source.platform.value}:dm:{source.chat_id}"
        else:
            session_key_preview = f"agent:main:{source.platform.value}:dm"
        if session_key_preview in self._pending_approvals:
            user_text = event.text.strip().lower()
            if user_text in ("yes", "y", "approve", "ok", "go", "do it"):
                approval = self._pending_approvals.pop(session_key_preview)
                cmd = approval["command"]
                pattern_key = approval.get("pattern_key", "")
                logger.info("User approved dangerous command: %s...", cmd[:60])
                from tools.terminal_tool import terminal_tool
                from tools.approval import approve_session
                approve_session(session_key_preview, pattern_key)
                result = terminal_tool(command=cmd, force=True)
                return f"✅ Command approved and executed.\n\n```\n{result[:3500]}\n```"
            elif user_text in ("no", "n", "deny", "cancel", "nope"):
                self._pending_approvals.pop(session_key_preview)
                return "❌ Command denied."
            # If it's not clearly an approval/denial, fall through to normal processing
        
        turn_state = await self._prepare_agent_turn_state(source, event)
        session_entry = turn_state["session_entry"]
        session_key = turn_state["session_key"]
        context_prompt = turn_state["context_prompt"]
        history = turn_state["history"]
        message_payload = turn_state["message_payload"]
        message_text = turn_state["message_text"]
        
        # First-message onboarding -- only on the very first interaction ever
        if not history and not self.session_store.has_any_sessions():
            context_prompt += (
                "\n\n[System note: This is the user's very first message ever. "
                "Briefly introduce yourself and mention that /help shows available commands. "
                "Keep the introduction concise -- one or two sentences max.]"
            )
        
        # One-time prompt if no home channel is set for this platform
        if not history and source.platform and source.platform != Platform.LOCAL:
            platform_name = source.platform.value
            env_key = f"{platform_name.upper()}_HOME_CHANNEL"
            if not os.getenv(env_key):
                adapter = self.adapters.get(source.platform)
                if adapter:
                    await adapter.send(
                        source.chat_id,
                        f"📬 No home channel is set for {platform_name.title()}. "
                        f"A home channel is where Hermes delivers cron job results "
                        f"and cross-platform messages.\n\n"
                        f"Type /sethome to make this chat your home channel, "
                        f"or ignore to skip."
                    )

        try:
            delivery_state = self._new_delivery_state(source, event)
            # Emit agent:start hook
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "message": message_text[:500],
            }
            await self.hooks.emit("agent:start", hook_ctx)
            
            # Run the agent
            agent_result = await self._run_agent(
                message=message_payload,
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=session_entry.session_id,
                session_key=session_key,
                event=event,
                delivery_state=delivery_state,
            )
            
            response = agent_result.get("final_response", "")
            delivery_response = self._append_request_cost_summary(
                response=response,
                agent_result=agent_result,
                source=source,
            )
            setattr(event, "_response_chat_id", str(delivery_state.get("chat_id") or source.chat_id))
            setattr(event, "_response_reply_to_message_id", delivery_state.get("reply_to"))
            agent_messages = agent_result.get("messages", [])
            use_delivery_response = True
            if agent_result.get("deferred_pending_event"):
                setattr(event, "_response_handled", True)
            
            # Emit agent:end hook
            await self.hooks.emit("agent:end", {
                **hook_ctx,
                "response": (response or "")[:500],
            })
            
            # Check for pending process watchers (check_interval on background processes)
            try:
                from tools.process_registry import process_registry
                while process_registry.pending_watchers:
                    watcher = process_registry.pending_watchers.pop(0)
                    asyncio.create_task(self._run_process_watcher(watcher))
            except Exception as e:
                logger.error("Process watcher setup error: %s", e)

            # Check if the agent encountered a dangerous command needing approval
            try:
                from tools.approval import pop_pending
                pending = pop_pending(session_key)
                if pending:
                    self._pending_approvals[session_key] = pending
            except Exception as e:
                logger.debug("Failed to check pending approvals: %s", e)
            
            # Save the full conversation to the transcript, including tool calls.
            # This preserves the complete agent loop (tool_calls, tool results,
            # intermediate reasoning) so sessions can be resumed with full context
            # and transcripts are useful for debugging and training data.
            ts = datetime.now().isoformat()
            
            # If this is a fresh session (no history), write the full tool
            # definitions as the first entry so the transcript is self-describing
            # -- the same list of dicts sent as tools=[...] in the API request.
            if not history:
                tool_defs = agent_result.get("tools", [])
                self.session_store.append_to_transcript(
                    session_entry.session_id,
                    {
                        "role": "session_meta",
                        "tools": tool_defs or [],
                        "model": os.getenv("HERMES_MODEL", ""),
                        "platform": source.platform.value if source.platform else "",
                        "timestamp": ts,
                    }
                )
            
            # Find only the NEW messages from this turn (skip history we loaded)
            history_input_len = int(agent_result.get("history_input_len", len(history)))
            history_input_len = max(0, history_input_len)
            new_messages = (
                agent_messages[history_input_len:]
                if len(agent_messages) > history_input_len
                else []
            )
            tool_defs = agent_result.get("tools", []) or []

            main_channel_response = response
            transcript_notice = None
            fork_request = None
            if delivery_state.get("thread_result") and delivery_state.get("thread_source"):
                thread_result = delivery_state["thread_result"]
                thread_source = delivery_state["thread_source"]
                thread_mention = str(thread_result.get("thread_mention") or f"<#{thread_result['thread_id']}>")
                transcript_notice = str(
                    delivery_state.get("transcript_notice") or f"[Continued in thread {thread_mention}]"
                )
                setattr(event, "_response_chat_id", str(delivery_state.get("chat_id") or thread_result["thread_id"]))
                setattr(event, "_response_reply_to_message_id", delivery_state.get("reply_to"))
                new_messages = self._rewrite_transcript_for_fork_notice(
                    new_messages,
                    transcript_notice=transcript_notice,
                    thread_mention=thread_mention,
                )
                if response and not delivery_state.get("thread_transcript_recorded"):
                    self._append_thread_fork_transcript(
                        thread_source=thread_source,
                        original_source=source,
                        request_text=str(getattr(event, "text", "") or message_text),
                        response=response,
                        tool_defs=tool_defs,
                    )
                    delivery_state["thread_transcript_recorded"] = True
                main_channel_response = response or main_channel_response
            else:
                fork_request = self._extract_fork_thread_request(new_messages)
            if response and not delivery_state.get("thread_result") and fork_request:
                auto_fork = await self._route_auto_fork_response(
                    event=event,
                    source=source,
                    message_text=message_text,
                    response=response,
                    delivery_response=delivery_response,
                    tool_defs=tool_defs,
                    fork_request=fork_request,
                )
                if auto_fork:
                    setattr(event, "_response_handled", True)
                    transcript_notice = auto_fork["transcript_notice"]
                    new_messages = self._rewrite_transcript_for_fork_notice(
                        new_messages,
                        transcript_notice=transcript_notice,
                        thread_mention=str(auto_fork["thread_result"]["thread_mention"]),
                    )
                    main_channel_response = auto_fork["main_notice"]
                    use_delivery_response = False
            
            # If no new messages found (edge case), fall back to simple user/assistant
            if not new_messages:
                self.session_store.append_to_transcript(
                    session_entry.session_id,
                    {"role": "user", "content": message_text, "timestamp": ts}
                )
                assistant_text = transcript_notice or main_channel_response or response
                if assistant_text:
                    self.session_store.append_to_transcript(
                        session_entry.session_id,
                        {"role": "assistant", "content": assistant_text, "timestamp": ts}
                    )
            else:
                for msg in new_messages:
                    # Skip system messages (they're rebuilt each run)
                    if msg.get("role") == "system":
                        continue
                    # Add timestamp to each message for debugging
                    entry = {**msg, "timestamp": ts}
                    self.session_store.append_to_transcript(
                        session_entry.session_id, entry
                    )
            
            # Update session
            request_usage = agent_result.get("request_usage") or {}
            input_tokens = int(request_usage.get("prompt_tokens", 0) or 0)
            output_tokens = int(request_usage.get("completion_tokens", 0) or 0)
            self.session_store.update_session(
                session_entry.session_key,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            
            return delivery_response if use_delivery_response else main_channel_response
            
        except Exception as e:
            logger.exception("Agent error in session %s", session_key)
            return (
                "Sorry, I encountered an unexpected error. "
                "The details have been logged for debugging. "
                "Try again or use /reset to start a fresh session."
            )
        finally:
            # Clear session env
            self._clear_session_env()
    
    async def _handle_reset_command(self, event: MessageEvent) -> str:
        """Handle /new or /reset command."""
        source = event.source
        
        # Get existing session key
        session_key = f"agent:main:{source.platform.value}:" + \
                      (f"dm" if source.chat_type == "dm" else f"{source.chat_type}:{source.chat_id}")
        
        # Memory flush before reset: load the old transcript and let a
        # temporary agent save memories before the session is wiped.
        try:
            old_entry = self.session_store._sessions.get(session_key)
            if old_entry:
                old_history = self.session_store.load_transcript(old_entry.session_id)
                if old_history:
                    from run_agent import AIAgent
                    loop = asyncio.get_event_loop()
                    # Resolve credentials so the flush agent can reach the LLM
                    _flush_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
                    _flush_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                    _flush_model = os.getenv("HERMES_MODEL") or os.getenv("LLM_MODEL", "anthropic/claude-opus-4.6")
                    def _do_flush():
                        tmp_agent = AIAgent(
                            model=_flush_model,
                            api_key=_flush_api_key,
                            base_url=_flush_base_url,
                            max_iterations=5,
                            quiet_mode=True,
                            enabled_toolsets=["memory"],
                            session_id=old_entry.session_id,
                            context_cwd=self._context_cwd,
                        )
                        # Build simple message list from transcript
                        msgs = []
                        for m in old_history:
                            role = m.get("role")
                            content = m.get("content")
                            if role in ("user", "assistant") and content:
                                msgs.append({"role": role, "content": content})
                        tmp_agent.flush_memories(msgs)
                    await loop.run_in_executor(None, _do_flush)
        except Exception as e:
            logger.debug("Gateway memory flush on reset failed: %s", e)
        
        # Reset the session
        new_entry = self.session_store.reset_session(session_key)

        # For Discord, reset channel-level turn anchoring so the next user turn
        # always starts from a fresh latest-N context window.
        if source.platform == Platform.DISCORD:
            adapter = self.adapters.get(Platform.DISCORD)
            reset_channel_context = getattr(adapter, "reset_channel_context", None)
            if callable(reset_channel_context):
                try:
                    reset_channel_context(source.chat_id)
                except Exception as e:
                    logger.debug("Discord channel context reset failed: %s", e)
        
        # Emit session:reset hook
        await self.hooks.emit("session:reset", {
            "platform": source.platform.value if source.platform else "",
            "user_id": source.user_id,
            "session_key": session_key,
        })
        
        if new_entry:
            return "✨ Session reset! I've started fresh with no memory of our previous conversation."
        else:
            # No existing session, just create one
            self.session_store.get_or_create_session(source, force_new=True)
            return "✨ New session started!"
    
    async def _handle_status_command(self, event: MessageEvent) -> str:
        """Handle /status command."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        
        connected_platforms = [p.value for p in self._connected_platforms()]
        
        # Check if there's an active agent
        session_key = session_entry.session_key
        is_running = session_key in self._running_agents
        
        lines = [
            "📊 **Hermes Gateway Status**",
            "",
            f"**Session ID:** `{session_entry.session_id[:12]}...`",
            f"**Created:** {session_entry.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Last Activity:** {session_entry.updated_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Tokens:** {session_entry.total_tokens:,}",
            f"**Agent Running:** {'Yes ⚡' if is_running else 'No'}",
            "",
            f"**Connected Platforms:** {', '.join(connected_platforms)}",
        ]
        
        return "\n".join(lines)

    async def _handle_fork_command(self, event: MessageEvent) -> str:
        """Handle /fork command for Discord threads."""
        source = event.source
        if source.platform != Platform.DISCORD:
            return "Fork is only available on Discord."

        adapter = self._live_discord_adapter()
        if adapter is None:
            return "Fork failed: Discord adapter is not available."

        creator = getattr(adapter, "create_fork_thread", None)
        if not callable(creator):
            return "Fork failed: this Discord adapter build does not support /fork."

        try:
            return await creator(event, event.get_command_args())
        except Exception as e:
            logger.debug("Discord /fork command failed: %s", e)
            return f"Fork failed: {e}"

    def _discord_auto_fork_prompt_note(self, source: SessionSource) -> str:
        """Return Discord auto-fork guidance when the tool is available."""
        if not self._discord_fork_available(source):
            return ""

        return (
            "[Discord routing note: Keep answers in the main channel by default. "
            "Use the `fork_thread` tool only when the discussion mainly benefits "
            "one person or is likely to become a long side thread that would "
            "lower channel signal-to-noise. After calling `fork_thread`, write "
            "your final response normally for the new thread. The gateway will "
            "post the substantive answer in that thread and leave a short "
            "handoff note in the main channel if configured.]"
        )

    @staticmethod
    def _discord_request_summary(
        source: SessionSource,
        request_text: str,
        *,
        reply_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a compact per-turn Discord summary for the active speaker."""
        current_user = str(source.user_name or source.user_id or "unknown").strip() or "unknown"
        request = str(request_text or "").strip()
        lines: List[str] = []
        if reply_context:
            reply_author = str(
                reply_context.get("author_display")
                or reply_context.get("author_name")
                or reply_context.get("author_id")
                or "unknown"
            ).strip() or "unknown"
            reply_preview = str(reply_context.get("preview") or "").strip()
            if reply_preview:
                lines.append(f'**Replying to {reply_author}**: "{reply_preview}"')
            else:
                lines.append(f"**Replying to {reply_author}**")
        lines.append(f"**Current user**: {current_user}")
        if "\n" in request:
            lines.append("**Request**:")
            lines.append(request)
            return "\n".join(lines)
        lines.append(f"**Request**: {request}")
        return "\n".join(lines)

    @staticmethod
    def _extract_fork_thread_request(messages: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """Extract the most recent successful fork_thread tool request from new messages."""
        tool_names: Dict[str, str] = {}
        request: Optional[Dict[str, str]] = None

        for msg in messages:
            if msg.get("role") == "assistant":
                for call in msg.get("tool_calls", []) or []:
                    call_id = str(call.get("id") or "").strip()
                    function = call.get("function") or {}
                    function_name = str(function.get("name") or "").strip()
                    if call_id and function_name:
                        tool_names[call_id] = function_name
                continue

            if msg.get("role") != "tool":
                continue

            tool_call_id = str(msg.get("tool_call_id") or "").strip()
            if not tool_call_id or tool_names.get(tool_call_id) != "fork_thread":
                continue

            try:
                payload = json.loads(msg.get("content") or "{}")
            except Exception:
                continue

            if not payload.get("success") or not payload.get("requested"):
                continue

            request = {
                "title": str(payload.get("title") or "").strip(),
                "visibility": str(payload.get("visibility") or "auto").strip().lower() or "auto",
                "reason": str(payload.get("reason") or "").strip(),
            }

        return request

    @staticmethod
    def _build_parent_fork_session_note(
        *,
        thread_mention: str,
        title: str = "",
        original_request: str = "",
    ) -> str:
        """Build a compact parent-session note for an already-forked topic."""
        thread_text = str(thread_mention or "").strip() or "the existing thread"
        title_text = " ".join(str(title or "").split()).strip()
        request_text = " ".join(str(original_request or "").split()).strip()
        if len(request_text) > 180:
            request_text = request_text[:177].rstrip() + "..."

        parts = [
            f"[Routing note: This side discussion was already forked to {thread_text}.",
        ]
        if title_text:
            parts.append(f"Thread topic: {title_text}.")
        if request_text:
            parts.append(f'Original request: "{request_text}".')
        parts.append(
            "If the user continues the same topic in the parent channel, do not answer it here; tell them to continue in the existing thread.]"
        )
        return " ".join(parts)

    @classmethod
    def _collapse_trailing_fork_handoffs(
        cls,
        history: List[Dict[str, Any]],
        *,
        source: Optional[SessionSource] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collapse trailing Discord fork handoff turns into one routing note.

        The persisted parent-channel transcript intentionally records the fork
        request and handoff note, but replaying the raw tool traffic into the
        next main-channel turn is noisy. Preserve only the useful state: that a
        thread already exists and repeated parent-channel follow-ups should be
        redirected there. Only collapse this pattern for Discord server channels.
        """
        if not history:
            return history

        if (
            source is None
            or source.platform != Platform.DISCORD
            or source.chat_type not in {"group", "channel"}
        ):
            return history

        collapsed = list(history)

        while len(collapsed) >= 4:
            notice_msg = collapsed[-1]
            tool_msg = collapsed[-2]
            assistant_msg = collapsed[-3]

            notice_text = notice_msg.get("content")
            if (
                notice_msg.get("role") != "assistant"
                or not isinstance(notice_text, str)
                or not notice_text.startswith("[Continued in thread <#")
                or not notice_text.endswith(">]")
            ):
                break

            if tool_msg.get("role") != "tool":
                break

            try:
                payload = json.loads(tool_msg.get("content") or "{}")
            except Exception:
                break

            if not payload.get("success") or not payload.get("requested"):
                break

            thread_id = str(payload.get("thread_id") or "").strip()
            if not thread_id:
                break

            if assistant_msg.get("role") != "assistant":
                break

            fork_call_ids = {
                str(call.get("id") or "").strip()
                for call in (assistant_msg.get("tool_calls") or [])
                if str((call.get("function") or {}).get("name") or "").strip() == "fork_thread"
            }
            if not fork_call_ids:
                break

            tool_call_id = str(tool_msg.get("tool_call_id") or "").strip()
            if tool_call_id and tool_call_id not in fork_call_ids:
                break

            start_index = len(collapsed) - 3
            prior_user_msg: Optional[Dict[str, Any]] = None
            if collapsed[start_index - 1].get("role") == "user":
                prior_user_msg = collapsed[start_index - 1]
                start_index -= 1

            thread_mention = str(payload.get("thread_mention") or f"<#{thread_id}>")
            parent_note = str(payload.get("parent_session_note") or "").strip()
            if not parent_note:
                original_request = ""
                if prior_user_msg is not None and isinstance(prior_user_msg.get("content"), str):
                    original_request = str(prior_user_msg.get("content") or "")
                parent_note = cls._build_parent_fork_session_note(
                    thread_mention=thread_mention,
                    title=str(payload.get("title") or ""),
                    original_request=original_request,
                )

            collapsed = collapsed[:start_index] + [
                {
                    "role": "assistant",
                    "content": parent_note,
                    "forked_to_thread": thread_mention,
                }
            ]
            break

        return collapsed

    @staticmethod
    def _rewrite_transcript_for_fork_notice(
        new_messages: List[Dict[str, Any]],
        transcript_notice: str,
        thread_mention: str,
    ) -> List[Dict[str, Any]]:
        """Replace the final assistant text with a short thread-handoff note."""
        rewritten = [dict(msg) for msg in new_messages]

        for index in range(len(rewritten) - 1, -1, -1):
            msg = rewritten[index]
            if msg.get("role") != "assistant" or "tool_calls" in msg:
                continue
            content = msg.get("content")
            if isinstance(content, str):
                updated = dict(msg)
                updated["content"] = transcript_notice
                updated["forked_to_thread"] = thread_mention
                rewritten[index] = updated
                return rewritten

        rewritten.append(
            {
                "role": "assistant",
                "content": transcript_notice,
                "forked_to_thread": thread_mention,
            }
        )
        return rewritten

    @staticmethod
    def _build_forked_user_transcript_content(
        original_source: SessionSource,
        request_text: str,
    ) -> str:
        """Render the user turn recorded in a forked thread transcript."""
        origin_label = original_source.chat_name or original_source.chat_id
        user_content = str(request_text or "").strip()
        if origin_label:
            user_content = f"[Forked from {origin_label}] {user_content}".strip()
        return user_content

    def _ensure_thread_fork_transcript_seeded(
        self,
        *,
        thread_source: SessionSource,
        original_source: SessionSource,
        request_text: str,
        tool_defs: List[Dict[str, Any]],
    ) -> SessionEntry:
        """Ensure the forked thread transcript has session metadata and the user turn."""
        thread_entry = self.session_store.get_or_create_session(thread_source)
        thread_history = self.session_store.load_transcript(thread_entry.session_id)
        ts = datetime.now().isoformat()

        if not thread_history:
            self.session_store.append_to_transcript(
                thread_entry.session_id,
                {
                    "role": "session_meta",
                    "tools": tool_defs or [],
                    "model": os.getenv("HERMES_MODEL", ""),
                    "platform": thread_source.platform.value if thread_source.platform else "",
                    "timestamp": ts,
                },
            )
            thread_history = self.session_store.load_transcript(thread_entry.session_id)

        user_content = self._build_forked_user_transcript_content(
            original_source,
            request_text,
        )
        has_user_turn = any(
            msg.get("role") == "user"
            and str(msg.get("content") or "") == user_content
            for msg in thread_history
        )
        if not has_user_turn:
            self.session_store.append_to_transcript(
                thread_entry.session_id,
                {"role": "user", "content": user_content, "timestamp": ts},
            )

        self.session_store.update_session(thread_entry.session_key)
        return thread_entry

    def _append_thread_fork_transcript(
        self,
        *,
        thread_source: SessionSource,
        original_source: SessionSource,
        request_text: str,
        response: str,
        tool_defs: List[Dict[str, Any]],
    ) -> None:
        """Seed the new thread session with the forked user turn and assistant reply."""
        thread_entry = self._ensure_thread_fork_transcript_seeded(
            thread_source=thread_source,
            original_source=original_source,
            request_text=request_text,
            tool_defs=tool_defs,
        )
        thread_history = self.session_store.load_transcript(thread_entry.session_id)
        last_non_meta = next(
            (msg for msg in reversed(thread_history) if msg.get("role") != "session_meta"),
            None,
        )
        if (
            last_non_meta is not None
            and last_non_meta.get("role") == "assistant"
            and str(last_non_meta.get("content") or "") == str(response or "")
        ):
            return

        ts = datetime.now().isoformat()
        self.session_store.append_to_transcript(
            thread_entry.session_id,
            {"role": "assistant", "content": response, "timestamp": ts},
        )
        self.session_store.update_session(thread_entry.session_key)

    @staticmethod
    def _build_thread_source(
        source: SessionSource,
        thread_result: Dict[str, Any],
    ) -> SessionSource:
        thread_id = str(thread_result.get("thread_id") or "").strip()
        return SessionSource(
            platform=Platform.DISCORD,
            chat_id=thread_id,
            chat_name=str(thread_result.get("thread_name") or thread_id),
            chat_type="thread",
            user_id=source.user_id,
            user_name=source.user_name,
            thread_id=thread_id,
        )

    async def _activate_live_fork_thread(
        self,
        *,
        event: MessageEvent,
        source: SessionSource,
        delivery_state: Dict[str, Any],
        title: str,
        visibility: str,
        reason: str,
        request_text: str = "",
        tool_defs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create the fork thread immediately and switch live delivery to it."""
        if delivery_state.get("thread_result"):
            result = dict(delivery_state["thread_result"])
            thread_mention = str(result.get("thread_mention") or f"<#{result.get('thread_id', '')}>")
            result.update(
                {
                    "success": True,
                    "requested": True,
                    "already_forked": True,
                    "reason": reason,
                    "parent_session_note": self._build_parent_fork_session_note(
                        thread_mention=thread_mention,
                        title=str(result.get("thread_name") or title or ""),
                        original_request=str(getattr(event, "text", "") or ""),
                    ),
                }
            )
            return result

        if source.platform != Platform.DISCORD or source.chat_type not in {"group", "channel"}:
            return {"success": False, "error": "fork_thread is only available in Discord server channels."}

        adapter = self._live_discord_adapter()
        if not adapter:
            return {"success": False, "error": "Discord adapter is not available."}

        checker = getattr(adapter, "is_auto_fork_available", None)
        if not callable(checker) or not checker(source):
            return {"success": False, "error": "Automatic thread forking is not enabled for this channel."}

        creator = getattr(adapter, "create_fork_thread_result", None)
        if not callable(creator):
            return {"success": False, "error": "This Discord adapter build does not support thread forking."}

        try:
            thread_result = await creator(
                event,
                requested_name=title or "",
                visibility=visibility or "auto",
            )
        except Exception as e:
            logger.debug("Discord live fork thread creation failed: %s", e)
            return {"success": False, "error": str(e)}

        if not thread_result.get("success"):
            return {
                "success": False,
                "error": str(thread_result.get("error") or "Thread creation failed."),
            }

        thread_source = self._build_thread_source(source, thread_result)
        thread_entry = self.session_store.get_or_create_session(thread_source)
        self._ensure_thread_fork_transcript_seeded(
            thread_source=thread_source,
            original_source=source,
            request_text=request_text or str(getattr(event, "text", "") or ""),
            tool_defs=tool_defs or [],
        )

        delivery_state["thread_result"] = thread_result
        delivery_state["thread_source"] = thread_source
        delivery_state["thread_session_id"] = thread_entry.session_id
        delivery_state["thread_session_key"] = thread_entry.session_key
        thread_chat_id = str(thread_result["thread_id"])
        delivery_state["chat_id"] = thread_chat_id
        delivery_state["reply_to"] = None
        thread_mention = str(thread_result.get("thread_mention") or f"<#{thread_result['thread_id']}>")
        delivery_state["transcript_notice"] = f"[Continued in thread {thread_mention}]"
        setattr(event, "_response_chat_id", thread_chat_id)
        setattr(event, "_response_reply_to_message_id", None)

        active_aliases = list(getattr(event, "_active_session_aliases", []) or [])
        if thread_chat_id not in active_aliases:
            active_aliases.append(thread_chat_id)
            setattr(event, "_active_session_aliases", active_aliases)

        active_sessions = getattr(adapter, "_active_sessions", None)
        if isinstance(active_sessions, dict):
            original_event = active_sessions.get(source.chat_id)
            if original_event is not None:
                active_sessions[thread_chat_id] = original_event

        notice_enabled = getattr(adapter, "auto_fork_main_channel_notice_enabled", None)
        main_notice = None
        if not callable(notice_enabled) or notice_enabled():
            main_notice = f"Taking this to a thread: {thread_mention}"
            try:
                await adapter.send(
                    chat_id=source.chat_id,
                    content=main_notice,
                    reply_to=event.message_id,
                )
                delivery_state["main_notice_sent"] = True
            except Exception as e:
                logger.debug("Discord live fork main-channel notice failed: %s", e)
        delivery_state["main_notice"] = main_notice

        payload = dict(thread_result)
        payload.update(
            {
                "success": True,
                "requested": True,
                "title": title,
                "visibility": str(thread_result.get("visibility") or visibility or "auto"),
                "reason": reason,
                "redirect_final_response": True,
                "main_channel_notice": main_notice is not None,
                "parent_session_note": self._build_parent_fork_session_note(
                    thread_mention=thread_mention,
                    title=str(thread_result.get("thread_name") or title or ""),
                    original_request=str(getattr(event, "text", "") or ""),
                ),
            }
        )
        return payload

    async def _route_auto_fork_response(
        self,
        *,
        event: MessageEvent,
        source: SessionSource,
        message_text: str,
        response: str,
        delivery_response: str,
        tool_defs: List[Dict[str, Any]],
        fork_request: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Create a Discord thread, deliver the full response there, and return notice metadata."""
        if source.platform != Platform.DISCORD or source.chat_type not in {"group", "channel"}:
            return None

        adapter = self._live_discord_adapter()
        if not adapter:
            return None

        checker = getattr(adapter, "is_auto_fork_available", None)
        if not callable(checker) or not checker(source):
            return None

        creator = getattr(adapter, "create_fork_thread_result", None)
        if not callable(creator):
            return None

        try:
            thread_result = await creator(
                event,
                requested_name=fork_request.get("title", ""),
                visibility=fork_request.get("visibility", "auto"),
            )
        except Exception as e:
            logger.debug("Discord auto-fork thread creation failed: %s", e)
            return None

        if not thread_result.get("success"):
            logger.debug("Discord auto-fork rejected: %s", thread_result.get("error"))
            return None

        delivered = await adapter.deliver_response(
            chat_id=str(thread_result["thread_id"]),
            response=delivery_response or response,
        )
        if not delivered:
            logger.warning(
                "Discord auto-fork created thread %s but failed to deliver the response there",
                thread_result.get("thread_id"),
            )
            return None

        thread_source = SessionSource(
            platform=Platform.DISCORD,
            chat_id=str(thread_result["thread_id"]),
            chat_name=str(thread_result.get("thread_name") or thread_result["thread_id"]),
            chat_type="thread",
            user_id=source.user_id,
            user_name=source.user_name,
            thread_id=str(thread_result["thread_id"]),
        )
        self._append_thread_fork_transcript(
            thread_source=thread_source,
            original_source=source,
            request_text=str(getattr(event, "text", "") or message_text),
            response=response,
            tool_defs=tool_defs,
        )

        thread_mention = str(thread_result.get("thread_mention") or f"<#{thread_result['thread_id']}>")
        transcript_notice = f"[Continued in thread {thread_mention}]"

        notice_enabled = getattr(adapter, "auto_fork_main_channel_notice_enabled", None)
        main_notice = None
        if not callable(notice_enabled) or notice_enabled():
            main_notice = f"Taking this to a thread: {thread_mention}"

        return {
            "thread_result": thread_result,
            "transcript_notice": transcript_notice,
            "main_notice": main_notice,
        }
    
    async def _handle_stop_command(self, event: MessageEvent) -> str:
        """Handle /stop command - interrupt a running agent."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key
        
        if session_key in self._running_agents:
            agent = self._running_agents[session_key]
            agent.interrupt()
            return "⚡ Stopping the current task... The agent will finish its current step and respond."
        else:
            return "No active task to stop."
    
    async def _handle_help_command(self, event: MessageEvent) -> str:
        """Handle /help command - list available commands."""
        return (
            "📖 **Hermes Commands**\n"
            "\n"
            "`/new` — Start a new conversation\n"
            "`/reset` — Reset conversation history\n"
            "`/status` — Show session info\n"
            "`/stop` — Interrupt the running agent\n"
            "`/model [name]` — Show or change the model\n"
            "`/personality [name]` — Set a personality\n"
            "`/retry` — Retry your last message\n"
            "`/undo` — Remove the last exchange\n"
            "`/fork [public|private] [name]` — Fork this Discord chat into a new thread (public by default)\n"
            "`/sethome` — Set this chat as the home channel\n"
            "`/help` — Show this message"
        )
    
    async def _handle_model_command(self, event: MessageEvent) -> str:
        """Handle /model command - show or change the current model."""
        import yaml

        args = event.get_command_args().strip()
        config_path = _hermes_home / 'config.yaml'

        # Resolve current model the same way the agent init does:
        # env vars first, then config.yaml always overrides.
        current = os.getenv("HERMES_MODEL") or os.getenv("LLM_MODEL") or "anthropic/claude-opus-4.6"
        try:
            if config_path.exists():
                with open(config_path) as f:
                    cfg = yaml.safe_load(f) or {}
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, str):
                    current = model_cfg
                elif isinstance(model_cfg, dict):
                    current = model_cfg.get("default", current)
        except Exception:
            pass

        if not args:
            return f"🤖 **Current model:** `{current}`\n\nTo change: `/model provider/model-name`"

        if "/" not in args:
            return (
                f"🤖 Invalid model format: `{args}`\n\n"
                f"Use `provider/model-name` format, e.g.:\n"
                f"• `anthropic/claude-sonnet-4`\n"
                f"• `google/gemini-2.5-pro`\n"
                f"• `openai/gpt-4o`"
            )

        # Write to config.yaml (source of truth), same pattern as CLI save_config_value.
        try:
            user_config = {}
            if config_path.exists():
                with open(config_path) as f:
                    user_config = yaml.safe_load(f) or {}
            if "model" not in user_config or not isinstance(user_config["model"], dict):
                user_config["model"] = {}
            user_config["model"]["default"] = args
            with open(config_path, 'w') as f:
                yaml.dump(user_config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            return f"⚠️ Failed to save model change: {e}"

        # Also set env var so code reading it before the next agent init sees the update.
        os.environ["HERMES_MODEL"] = args

        return f"🤖 Model changed to `{args}`\n_(takes effect on next message)_"
    
    async def _handle_personality_command(self, event: MessageEvent) -> str:
        """Handle /personality command - list or set a personality."""
        import yaml

        args = event.get_command_args().strip().lower()
        config_path = _hermes_home / 'config.yaml'

        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                personalities = config.get("agent", {}).get("personalities", {})
            else:
                config = {}
                personalities = {}
        except Exception:
            config = {}
            personalities = {}

        if not personalities:
            return "No personalities configured in `~/.hermes/config.yaml`"

        if not args:
            lines = ["🎭 **Available Personalities**\n"]
            for name, prompt in personalities.items():
                preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
                lines.append(f"• `{name}` — {preview}")
            lines.append(f"\nUsage: `/personality <name>`")
            return "\n".join(lines)

        if args in personalities:
            new_prompt = personalities[args]

            # Write to config.yaml, same pattern as CLI save_config_value.
            try:
                if "agent" not in config or not isinstance(config.get("agent"), dict):
                    config["agent"] = {}
                config["agent"]["system_prompt"] = new_prompt
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except Exception as e:
                return f"⚠️ Failed to save personality change: {e}"

            # Update in-memory so it takes effect on the very next message.
            self._ephemeral_system_prompt = new_prompt

            return f"🎭 Personality set to **{args}**\n_(takes effect on next message)_"

        available = ", ".join(f"`{n}`" for n in personalities.keys())
        return f"Unknown personality: `{args}`\n\nAvailable: {available}"
    
    async def _handle_retry_command(self, event: MessageEvent) -> str:
        """Handle /retry command - re-send the last user message."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        
        # Find the last user message
        last_user_msg = None
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_msg = history[i].get("content", "")
                last_user_idx = i
                break
        
        if not last_user_msg:
            return "No previous message to retry."
        
        # Truncate history to before the last user message
        truncated = history[:last_user_idx]
        session_entry.conversation_history = truncated
        
        # Re-send by creating a fake text event with the old message
        retry_event = MessageEvent(
            text=last_user_msg,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event.raw_message,
        )
        
        # Let the normal message handler process it
        await self._handle_message(retry_event)
        return None  # Response sent through normal flow
    
    async def _handle_undo_command(self, event: MessageEvent) -> str:
        """Handle /undo command - remove the last user/assistant exchange."""
        source = event.source
        session_entry = self.session_store.get_or_create_session(source)
        history = self.session_store.load_transcript(session_entry.session_id)
        
        # Find the last user message and remove everything from it onward
        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_idx = i
                break
        
        if last_user_idx is None:
            return "Nothing to undo."
        
        removed_msg = history[last_user_idx].get("content", "")
        removed_count = len(history) - last_user_idx
        session_entry.conversation_history = history[:last_user_idx]
        
        preview = removed_msg[:40] + "..." if len(removed_msg) > 40 else removed_msg
        return f"↩️ Undid {removed_count} message(s).\nRemoved: \"{preview}\""
    
    async def _handle_set_home_command(self, event: MessageEvent) -> str:
        """Handle /sethome command -- set the current chat as the platform's home channel."""
        source = event.source
        platform_name = source.platform.value if source.platform else "unknown"
        chat_id = source.chat_id
        chat_name = source.chat_name or chat_id
        
        env_key = f"{platform_name.upper()}_HOME_CHANNEL"
        
        # Save to config.yaml
        try:
            import yaml
            config_path = _hermes_home / 'config.yaml'
            user_config = {}
            if config_path.exists():
                with open(config_path) as f:
                    user_config = yaml.safe_load(f) or {}
            user_config[env_key] = chat_id
            with open(config_path, 'w') as f:
                yaml.dump(user_config, f, default_flow_style=False)
            # Also set in the current environment so it takes effect immediately
            os.environ[env_key] = str(chat_id)
        except Exception as e:
            return f"Failed to save home channel: {e}"
        
        return (
            f"✅ Home channel set to **{chat_name}** (ID: {chat_id}).\n"
            f"Cron jobs and cross-platform messages will be delivered here."
        )
    
    def _set_session_env(self, context: SessionContext) -> None:
        """Set environment variables for the current session."""
        os.environ["HERMES_SESSION_PLATFORM"] = context.source.platform.value
        os.environ["HERMES_SESSION_CHAT_ID"] = context.source.chat_id
        os.environ["HERMES_SESSION_CHAT_TYPE"] = context.source.chat_type
        os.environ["HERMES_DISCORD_FORK_THREAD_AVAILABLE"] = (
            "1" if self._discord_fork_available(context.source) else "0"
        )
        if context.source.chat_name:
            os.environ["HERMES_SESSION_CHAT_NAME"] = context.source.chat_name
        if context.source.thread_id:
            os.environ["HERMES_SESSION_THREAD_ID"] = context.source.thread_id

    def _clear_session_env(self) -> None:
        """Clear session environment variables."""
        for var in [
            "HERMES_SESSION_PLATFORM",
            "HERMES_SESSION_CHAT_ID",
            "HERMES_SESSION_CHAT_TYPE",
            "HERMES_DISCORD_FORK_THREAD_AVAILABLE",
            "HERMES_SESSION_CHAT_NAME",
            "HERMES_SESSION_THREAD_ID",
        ]:
            if var in os.environ:
                del os.environ[var]

    def _current_model_name(self) -> str:
        """Resolve the active model name using env + ~/.hermes/config.yaml."""
        model = os.getenv("HERMES_MODEL") or os.getenv("LLM_MODEL") or "anthropic/claude-opus-4.6"
        try:
            import yaml
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, str):
                    model = model_cfg
                elif isinstance(model_cfg, dict):
                    model = model_cfg.get("default", model)
        except Exception:
            pass
        return str(model or "").strip()

    @staticmethod
    def _model_supports_direct_multimodal(model_name: str) -> bool:
        """Return True when the model family is known to support image_url input."""
        model = (model_name or "").strip().lower()
        if not model:
            return False
        if model.startswith(("openai/", "anthropic/", "google/gemini")):
            return True
        return model.startswith(("gpt-", "claude", "gemini"))

    @staticmethod
    def _is_http_url(value: str) -> bool:
        value = (value or "").strip().lower()
        return value.startswith("http://") or value.startswith("https://")

    @staticmethod
    def _is_data_image_url(value: str) -> bool:
        value = (value or "").strip().lower()
        return value.startswith("data:image/")

    def _normalize_data_image_url(self, value: str) -> str:
        """
        Normalize `data:image/...;base64,...` URLs so declared MIME matches bytes.

        Returns the original value when parsing/decoding is not possible.
        """
        data_url = str(value or "").strip()
        if not self._is_data_image_url(data_url):
            return data_url

        try:
            header, encoded = data_url.split(",", 1)
        except ValueError:
            return data_url

        lower_header = header.lower()
        if ";base64" not in lower_header:
            return data_url

        declared_mime = lower_header[5:].split(";", 1)[0].strip()
        cleaned_encoded = re.sub(r"\s+", "", encoded)
        try:
            raw = base64.b64decode(cleaned_encoded, validate=True)
        except Exception:
            return data_url

        resolved_mime = self._resolve_image_mime(
            raw=raw,
            declared_mime=declared_mime,
            name_hint="data-url",
        )
        if not resolved_mime or resolved_mime == declared_mime:
            return data_url

        logger.debug(
            "Normalized data URL image MIME mismatch: declared=%s, resolved=%s",
            declared_mime,
            resolved_mime,
        )
        return f"data:{resolved_mime};base64,{cleaned_encoded}"

    @staticmethod
    def _is_gif_media(path: str, media_type: str) -> bool:
        mtype = (media_type or "").split(";", 1)[0].strip().lower()
        if mtype == "image/gif":
            return True
        path_no_query = (path or "").split("?", 1)[0].lower()
        return path_no_query.endswith(".gif")

    @staticmethod
    def _sniff_image_mime(raw: bytes) -> Optional[str]:
        """Best-effort MIME detection from image file signatures."""
        if not raw:
            return None
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if raw.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if raw.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return "image/webp"
        if raw.startswith(b"BM"):
            return "image/bmp"
        if raw.startswith((b"II*\x00", b"MM\x00*")):
            return "image/tiff"
        if len(raw) >= 12 and raw[4:8] == b"ftyp":
            brand = raw[8:12]
            if brand in (b"avif", b"avis"):
                return "image/avif"
            if brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"):
                return "image/heic"
        return None

    @classmethod
    def _resolve_image_mime(
        cls,
        raw: bytes,
        declared_mime: str,
        name_hint: str,
    ) -> str:
        """Resolve a safe data-URL MIME, preferring sniffed bytes when available."""
        declared = str(declared_mime or "").split(";", 1)[0].strip().lower()
        if not declared.startswith("image/") or declared in ("image/*", "image/x-discord-emoji"):
            declared = ""

        sniffed = cls._sniff_image_mime(raw)
        if sniffed:
            if declared and declared != sniffed:
                logger.debug(
                    "Image MIME mismatch for %s: declared=%s, sniffed=%s; using sniffed",
                    name_hint,
                    declared,
                    sniffed,
                )
            return sniffed

        if declared:
            return declared

        guessed, _ = mimetypes.guess_type(name_hint)
        guessed = str(guessed or "").split(";", 1)[0].strip().lower()
        if guessed.startswith("image/") and guessed != "image/*":
            return guessed
        return "image/jpeg"

    @staticmethod
    def _emoji_name_from_media_type(media_type: str) -> str:
        for part in str(media_type or "").split(";"):
            part = part.strip()
            if part.startswith("name="):
                return part.split("=", 1)[1].strip()
        return ""

    @staticmethod
    def _source_url_from_media_type(media_type: str) -> str:
        for part in str(media_type or "").split(";"):
            part = part.strip()
            if part.startswith("source_url="):
                return part.split("=", 1)[1].strip()
        return ""

    def _collect_multimodal_image_items(self, event: MessageEvent) -> List[Dict[str, str]]:
        """
        Collect image inputs from an event for direct multimodal payloads.

        Filters out GIFs and captures optional emoji labels.
        """
        items: List[Dict[str, str]] = []
        seen_keys = set()
        attachments = list(getattr(getattr(event, "raw_message", None), "attachments", []) or [])
        for i, path in enumerate(event.media_urls or []):
            media_type = event.media_types[i] if i < len(event.media_types) else ""
            lower_type = str(media_type or "").lower()
            is_discord_emoji = lower_type.startswith("image/x-discord-emoji")
            is_image = lower_type.startswith("image/") or is_discord_emoji or event.message_type == MessageType.PHOTO
            if not is_image or self._is_gif_media(path, media_type):
                continue

            source_url = self._source_url_from_media_type(media_type)
            if not source_url and i < len(attachments):
                att = attachments[i]
                att_type = getattr(att, "content_type", "") or ""
                att_url = getattr(att, "url", "") or ""
                if att_type.startswith("image/") and not self._is_gif_media(att_url, att_type):
                    source_url = att_url

            dedupe_key = source_url or path
            if dedupe_key and dedupe_key in seen_keys:
                continue
            if dedupe_key:
                seen_keys.add(dedupe_key)
            items.append(
                {
                    "path": path,
                    "media_type": media_type or "image/*",
                    "source_url": source_url,
                    "emoji_name": self._emoji_name_from_media_type(media_type) if is_discord_emoji else "",
                }
            )

        # Optional follow-up image carryover (e.g., "image + next text turn").
        for extra in list(getattr(event, "carryover_image_items", []) or []):
            if not isinstance(extra, dict):
                continue
            path = (extra.get("path") or "").strip()
            source_url = (extra.get("source_url") or "").strip()
            media_type = (extra.get("media_type") or "image/*").strip() or "image/*"
            emoji_name = (extra.get("emoji_name") or "").strip()
            dedupe_key = source_url or path
            if not dedupe_key:
                continue
            if dedupe_key in seen_keys:
                continue
            if self._is_gif_media(dedupe_key, media_type):
                continue
            seen_keys.add(dedupe_key)
            items.append(
                {
                    "path": path or source_url,
                    "media_type": media_type,
                    "source_url": source_url,
                    "emoji_name": emoji_name,
                }
            )
        return items

    def _local_image_to_data_url(self, path: str, media_type: str) -> Optional[str]:
        """Convert a local image path into a data URL for API image_url fields."""
        if self._is_data_image_url(path):
            return str(path).strip()

        try:
            p = Path(path)
            if not p.exists() or not p.is_file():
                return None
        except (OSError, ValueError) as e:
            logger.debug("Could not interpret image path for multimodal payload (%r): %s", path, e)
            return None
        try:
            if p.stat().st_size > 8 * 1024 * 1024:
                logger.warning("Skipping oversized local image for multimodal payload: %s", p)
                return None
            raw = p.read_bytes()
            mime_type = self._resolve_image_mime(
                raw=raw,
                declared_mime=media_type,
                name_hint=p.name,
            )
            encoded = base64.b64encode(raw).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            logger.debug("Could not convert local image to data URL (%s): %s", p, e)
            return None

    def _http_image_to_data_url(self, url: str, media_type: str) -> Optional[str]:
        """Download an HTTP image URL and convert it to a base64 data URL."""
        if not self._is_http_url(url):
            return None
        try:
            max_bytes = 8 * 1024 * 1024
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)",
                    "Accept": "image/*,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    logger.warning("Skipping oversized remote image for multimodal payload: %s", url)
                    return None
                header_mime = str(resp.headers.get("Content-Type", "") or "").split(";", 1)[0].strip().lower()
            declared_mime = header_mime if header_mime.startswith("image/") else media_type
            mime_type = self._resolve_image_mime(
                raw=raw,
                declared_mime=declared_mime,
                name_hint=urllib.parse.urlparse(url).path,
            )
            encoded = base64.b64encode(raw).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            logger.debug("Could not convert remote image URL to data URL (%s): %s", url, e)
            return None

    def _resolve_image_payload_url(self, item: Dict[str, str]) -> Optional[str]:
        """Resolve an image URL usable by chat-completions image_url content parts."""
        path = (item.get("path") or "").strip()
        source_url = (item.get("source_url") or "").strip()

        # Normalize existing data URLs from history/inputs so MIME matches bytes.
        if self._is_data_image_url(source_url):
            return self._normalize_data_image_url(source_url)
        if self._is_data_image_url(path):
            return self._normalize_data_image_url(path)

        # Prefer local cached files so providers that reject remote URL sources
        # still receive a valid base64-encoded image payload.
        data_url = self._local_image_to_data_url(path, item.get("media_type", ""))
        if data_url:
            return data_url

        data_url = self._http_image_to_data_url(source_url, item.get("media_type", ""))
        if data_url:
            return data_url
        data_url = self._http_image_to_data_url(path, item.get("media_type", ""))
        if data_url:
            return data_url
        # Do not pass raw HTTP URLs through to providers that require base64 sources.
        return None

    def _sanitize_multimodal_history_content(self, content: Any) -> Any:
        """
        Ensure historical multimodal image parts are base64 data URLs.

        Older transcript rows may contain raw HTTP image_url values from before
        local/base64 normalization was added. Convert when possible; otherwise
        replace with a text placeholder so requests don't hard-fail.
        """
        if not isinstance(content, list):
            return content

        sanitized: List[Any] = []
        for part in content:
            if not isinstance(part, dict):
                sanitized.append(part)
                continue
            if part.get("type") != "image_url":
                sanitized.append(part)
                continue

            image_obj = part.get("image_url")
            url = image_obj.get("url", "") if isinstance(image_obj, dict) else ""
            payload_url = self._resolve_image_payload_url(
                {
                    "path": str(url or ""),
                    "source_url": str(url or ""),
                    "media_type": "",
                }
            )
            if payload_url:
                new_part = dict(part)
                new_image_obj = dict(image_obj) if isinstance(image_obj, dict) else {}
                new_image_obj["url"] = payload_url
                new_part["image_url"] = new_image_obj
                sanitized.append(new_part)
            else:
                sanitized.append({"type": "text", "text": "[image omitted: unavailable]"})

        return sanitized

    def _build_multimodal_user_payload(
        self,
        user_text: str,
        image_items: List[Dict[str, str]],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Build OpenAI-style user content parts with text + image_url entries.

        Emoji images are preceded by a short `<emoji_name>:` text part.
        """
        if not image_items:
            return None

        parts: List[Dict[str, Any]] = []
        text = (user_text or "").strip()
        if text:
            parts.append({"type": "text", "text": text})
        else:
            parts.append({"type": "text", "text": "[User attached image(s)]"})

        added_images = 0
        for idx, item in enumerate(image_items, start=1):
            payload_url = self._resolve_image_payload_url(item)
            if not payload_url:
                continue
            emoji_name = (item.get("emoji_name") or "").strip()
            if emoji_name:
                parts.append({"type": "text", "text": f"{emoji_name}: "})
            elif len(image_items) > 1:
                parts.append({"type": "text", "text": f"[attached_image_{idx}]"})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": payload_url},
                }
            )
            added_images += 1

        if added_images == 0:
            return None
        return parts

    async def _enrich_message_with_vision(
        self,
        user_text: str,
        image_paths: List[str],
    ) -> str:
        """
        Auto-analyze user-attached images with the vision tool and prepend
        the descriptions to the message text.

        Each image is analyzed with a general-purpose prompt.  The resulting
        description *and* the local cache path are injected so the model can:
          1. Immediately understand what the user sent (no extra tool call).
          2. Re-examine the image with vision_analyze if it needs more detail.

        Args:
            user_text:   The user's original caption / message text.
            image_paths: List of local file paths to cached images.

        Returns:
            The enriched message string with vision descriptions prepended.
        """
        from tools.vision_tools import vision_analyze_tool
        import json as _json

        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        enriched_parts = []
        for path in image_paths:
            try:
                logger.debug("Auto-analyzing user image: %s", path)
                result_json = await vision_analyze_tool(
                    image_url=path,
                    user_prompt=analysis_prompt,
                )
                result = _json.loads(result_json)
                if result.get("success"):
                    description = result.get("analysis", "")
                    enriched_parts.append(
                        f"[The user sent an image~ Here's what I can see:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {path} ~]"
                    )
                else:
                    enriched_parts.append(
                        "[The user sent an image but I couldn't quite see it "
                        "this time (>_<) You can try looking at it yourself "
                        f"with vision_analyze using image_url: {path}]"
                    )
            except Exception as e:
                logger.error("Vision auto-analysis error: %s", e)
                enriched_parts.append(
                    f"[The user sent an image but something went wrong when I "
                    f"tried to look at it~ You can try examining it yourself "
                    f"with vision_analyze using image_url: {path}]"
                )

        # Combine: vision descriptions first, then the user's original text
        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    async def _enrich_message_with_transcription(
        self,
        user_text: str,
        audio_paths: List[str],
    ) -> str:
        """
        Auto-transcribe user voice/audio messages using OpenAI Whisper API
        and prepend the transcript to the message text.

        Args:
            user_text:   The user's original caption / message text.
            audio_paths: List of local file paths to cached audio files.

        Returns:
            The enriched message string with transcriptions prepended.
        """
        from tools.transcription_tools import transcribe_audio
        import asyncio

        enriched_parts = []
        for path in audio_paths:
            try:
                logger.debug("Transcribing user voice: %s", path)
                result = await asyncio.to_thread(transcribe_audio, path)
                if result["success"]:
                    transcript = result["transcript"]
                    enriched_parts.append(
                        f'[The user sent a voice message~ '
                        f'Here\'s what they said: "{transcript}"]'
                    )
                else:
                    error = result.get("error", "unknown error")
                    if "OPENAI_API_KEY" in error or "VOICE_TOOLS_OPENAI_KEY" in error:
                        enriched_parts.append(
                            "[The user sent a voice message but I can't listen "
                            "to it right now~ VOICE_TOOLS_OPENAI_KEY isn't set up yet "
                            "(';w;') Let them know!]"
                        )
                    else:
                        enriched_parts.append(
                            "[The user sent a voice message but I had trouble "
                            f"transcribing it~ ({error})]"
                        )
            except Exception as e:
                logger.error("Transcription error: %s", e)
                enriched_parts.append(
                    "[The user sent a voice message but something went wrong "
                    "when I tried to listen to it~ Let them know!]"
                )

        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    async def _run_process_watcher(self, watcher: dict) -> None:
        """
        Periodically check a background process and push updates to the user.

        Runs as an asyncio task. Stays silent when nothing changed.
        Auto-removes when the process exits or is killed.
        """
        from tools.process_registry import process_registry

        session_id = watcher["session_id"]
        interval = watcher["check_interval"]
        session_key = watcher.get("session_key", "")
        platform_name = watcher.get("platform", "")
        chat_id = watcher.get("chat_id", "")

        logger.debug("Process watcher started: %s (every %ss)", session_id, interval)

        last_output_len = 0
        while True:
            await asyncio.sleep(interval)

            session = process_registry.get(session_id)
            if session is None:
                break

            current_output_len = len(session.output_buffer)
            has_new_output = current_output_len > last_output_len
            last_output_len = current_output_len

            if session.exited:
                # Process finished -- deliver final update
                new_output = session.output_buffer[-1000:] if session.output_buffer else ""
                message_text = (
                    f"[Background process {session_id} finished with exit code {session.exit_code}~ "
                    f"Here's the final output:\n{new_output}]"
                )
                # Try to deliver to the originating platform
                adapter = None
                for p, a in self.adapters.items():
                    if p.value == platform_name:
                        adapter = a
                        break
                if adapter and chat_id:
                    try:
                        await adapter.send(chat_id, message_text)
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)
                break

            elif has_new_output:
                # New output available -- deliver status update
                new_output = session.output_buffer[-500:] if session.output_buffer else ""
                message_text = (
                    f"[Background process {session_id} is still running~ "
                    f"New output:\n{new_output}]"
                )
                adapter = None
                for p, a in self.adapters.items():
                    if p.value == platform_name:
                        adapter = a
                        break
                if adapter and chat_id:
                    try:
                        await adapter.send(chat_id, message_text)
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)

        logger.debug("Process watcher ended: %s", session_id)

    async def _run_agent(
        self,
        message: Any,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None,
        event: Optional[MessageEvent] = None,
        delivery_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent with the given message and context.
        
        Returns the full result dict from run_conversation, including:
          - "final_response": str (the text to send back)
          - "messages": list (full conversation including tool calls)
          - "api_calls": int
          - "completed": bool
        
        This is run in a thread pool to not block the event loop.
        Supports interruption via new messages.
        """
        from run_agent import AIAgent
        import queue
        main_loop = asyncio.get_running_loop()

        if delivery_state is None:
            delivery_state = {
                "chat_id": source.chat_id,
                "reply_to": getattr(event, "message_id", None),
                "thread_result": None,
                "thread_source": None,
                "thread_session_id": None,
                "thread_session_key": None,
                "transcript_notice": None,
                "main_notice": None,
                "main_notice_sent": False,
                "thread_transcript_recorded": False,
            }
        
        # Determine toolset based on platform.
        # Check config.yaml for per-platform overrides, fallback to hardcoded defaults.
        default_toolset_map = {
            Platform.LOCAL: "hermes-cli",
            Platform.TELEGRAM: "hermes-telegram",
            Platform.DISCORD: "hermes-discord",
            Platform.WHATSAPP: "hermes-whatsapp",
            Platform.SLACK: "hermes-slack",
        }
        
        # Try to load platform_toolsets from config
        platform_toolsets_config = {}
        try:
            config_path = _hermes_home / 'config.yaml'
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                platform_toolsets_config = user_config.get("platform_toolsets", {})
        except Exception as e:
            logger.debug("Could not load platform_toolsets config: %s", e)
        
        # Map platform enum to config key
        platform_config_key = {
            Platform.LOCAL: "cli",
            Platform.TELEGRAM: "telegram",
            Platform.DISCORD: "discord",
            Platform.WHATSAPP: "whatsapp",
            Platform.SLACK: "slack",
        }.get(source.platform, "telegram")
        
        # Use config override if present (list of toolsets), otherwise hardcoded default
        config_toolsets = platform_toolsets_config.get(platform_config_key)
        if config_toolsets and isinstance(config_toolsets, list):
            enabled_toolsets = config_toolsets
        else:
            default_toolset = default_toolset_map.get(source.platform, "hermes-telegram")
            enabled_toolsets = [default_toolset]
        
        # Tool progress mode from config.yaml: "all", "new", "verbose", "off"
        # Falls back to env vars for backward compatibility
        _progress_cfg = {}
        try:
            _tp_cfg_path = _hermes_home / "config.yaml"
            if _tp_cfg_path.exists():
                import yaml as _tp_yaml
                with open(_tp_cfg_path) as _tp_f:
                    _tp_data = _tp_yaml.safe_load(_tp_f) or {}
                _progress_cfg = _tp_data.get("display", {})
        except Exception:
            pass
        progress_mode = (
            _progress_cfg.get("tool_progress")
            or os.getenv("HERMES_TOOL_PROGRESS_MODE")
            or "all"
        )
        tool_progress_enabled = progress_mode != "off"
        
        # Queue for progress messages (thread-safe)
        progress_queue = queue.Queue() if tool_progress_enabled else None
        last_tool = [None]  # Mutable container for tracking in closure
        
        def progress_callback(tool_name: str, preview: str = None):
            """Callback invoked by agent when a tool is called."""
            if not progress_queue:
                return
            if tool_name == "fork_thread":
                return
            
            # "new" mode: only report when tool changes
            if progress_mode == "new" and tool_name == last_tool[0]:
                return
            last_tool[0] = tool_name
            
            # Build progress message with primary argument preview
            tool_emojis = {
                "terminal": "💻",
                "process": "⚙️",
                "web_search": "🔍",
                "web_extract": "📄",
                "read_file": "📖",
                "write_file": "✍️",
                "patch": "🔧",
                "search": "🔎",
                "list_directory": "📂",
                "image_generate": "🎨",
                "text_to_speech": "🔊",
                "browser_navigate": "🌐",
                "browser_click": "👆",
                "browser_type": "⌨️",
                "browser_snapshot": "📸",
                "browser_scroll": "📜",
                "browser_back": "◀️",
                "browser_press": "⌨️",
                "browser_close": "🚪",
                "browser_get_images": "🖼️",
                "browser_vision": "👁️",
                "moa_query": "🧠",
                "mixture_of_agents": "🧠",
                "vision_analyze": "👁️",
                "skill_view": "📚",
                "skills_list": "📋",
                "todo": "📋",
                "memory": "🧠",
                "session_search": "🔍",
                "send_message": "📨",
                "fork_thread": "🧵",
                "schedule_cronjob": "⏰",
                "list_cronjobs": "⏰",
                "remove_cronjob": "⏰",
            }
            emoji = tool_emojis.get(tool_name, "⚙️")
            
            if preview:
                # Truncate preview to keep messages clean
                if len(preview) > 40:
                    preview = preview[:37] + "..."
                msg = f"{emoji} {tool_name}... \"{preview}\""
            else:
                msg = f"{emoji} {tool_name}..."
            
            progress_queue.put(msg)
        
        # Background task to send progress messages
        async def send_progress_messages():
            if not progress_queue:
                return
            
            adapter = self.adapters.get(source.platform)
            if not adapter:
                return
            
            while True:
                try:
                    # Non-blocking check with small timeout
                    msg = progress_queue.get_nowait()
                    target_chat_id = str(delivery_state.get("chat_id") or source.chat_id)
                    await adapter.send(chat_id=target_chat_id, content=msg)
                    # Restore typing indicator after sending progress message
                    await asyncio.sleep(0.3)
                    await adapter.send_typing(target_chat_id)
                except queue.Empty:
                    await asyncio.sleep(0.3)  # Check again soon
                except asyncio.CancelledError:
                    # Drain remaining messages
                    while not progress_queue.empty():
                        try:
                            msg = progress_queue.get_nowait()
                            target_chat_id = str(delivery_state.get("chat_id") or source.chat_id)
                            await adapter.send(chat_id=target_chat_id, content=msg)
                        except Exception:
                            break
                    return
                except Exception as e:
                    logger.error("Progress message error: %s", e)
                    await asyncio.sleep(1)
        
        # We need to share the agent instance for interrupt support
        agent_holder = [None]  # Mutable container for the agent instance
        result_holder = [None]  # Mutable container for the result
        tools_holder = [None]   # Mutable container for the tool definitions
        
        def run_sync():
            # Pass session_key to process registry via env var so background
            # processes can be mapped back to this gateway session
            os.environ["HERMES_SESSION_KEY"] = session_key or ""

            # Read from env var or use default (same as CLI)
            max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "60"))
            
            # Map platform enum to the platform hint key the agent understands.
            # Platform.LOCAL ("local") maps to "cli"; others pass through as-is.
            platform_key = "cli" if source.platform == Platform.LOCAL else source.platform.value
            
            # Combine platform context with user-configured ephemeral system prompt
            combined_ephemeral = context_prompt or ""
            if self._ephemeral_system_prompt:
                combined_ephemeral = (combined_ephemeral + "\n\n" + self._ephemeral_system_prompt).strip()
            
            # Re-read .env and config for fresh credentials (gateway is long-lived,
            # keys may change without restart).
            try:
                load_dotenv(_env_path, override=True, encoding="utf-8")
            except UnicodeDecodeError:
                load_dotenv(_env_path, override=True, encoding="latin-1")
            except Exception:
                pass

            # Keep messaging sessions pinned to MESSAGING_CWD. Re-loading .env
            # with override=True can re-introduce a stale TERMINAL_CWD (e.g. ".")
            # which breaks container backends that require absolute -w paths.
            os.environ["TERMINAL_CWD"] = os.getenv("MESSAGING_CWD") or str(Path.home())

            # Custom endpoint (OPENAI_*) takes precedence for base_url.
            # API key selection is based on the effective provider/base URL to
            # avoid sending OPENAI_API_KEY to OpenRouter endpoints.
            openai_key = os.getenv("OPENAI_API_KEY", "").strip()
            openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

            def _select_api_key(target_base_url: str, provider_hint: str = "") -> str:
                provider_hint = (provider_hint or "").strip().lower()
                target_base = (target_base_url or "").lower()
                if provider_hint == "openrouter" or "openrouter.ai" in target_base:
                    return openrouter_key or openai_key
                return openai_key or openrouter_key

            api_key = _select_api_key(base_url)
            model = os.getenv("HERMES_MODEL") or os.getenv("LLM_MODEL") or "anthropic/claude-opus-4.6"
            model_extra_body = None

            try:
                resolved = load_model_runtime_config(
                    _hermes_home / "config.yaml",
                    default_model=model,
                    default_base_url=base_url,
                    logger=logger,
                )
                model = resolved.model
                base_url = resolved.base_url
                model_extra_body = resolved.extra_body
                if resolved.provider == "nous":
                    try:
                        from hermes_cli.auth import resolve_nous_runtime_credentials
                        creds = resolve_nous_runtime_credentials(min_key_ttl_seconds=5 * 60)
                        api_key = creds.get("api_key", api_key)
                        base_url = creds.get("base_url", base_url)
                    except Exception as nous_err:
                        logger.warning("Nous Portal credential resolution failed: %s", nous_err)
                else:
                    # Re-evaluate key selection after config-based base_url/provider override.
                    api_key = _select_api_key(base_url, resolved.provider)
            except Exception:
                pass

            def fork_thread_callback(title: str = "", visibility: str = "auto", reason: str = "") -> Dict[str, Any]:
                if event is None:
                    return {"success": False, "error": "No source event available for thread creation."}

                future = asyncio.run_coroutine_threadsafe(
                    self._activate_live_fork_thread(
                        event=event,
                        source=source,
                        delivery_state=delivery_state,
                        title=title,
                        visibility=visibility,
                        reason=reason,
                        request_text=str(getattr(event, "text", "") or ""),
                        tool_defs=tools_holder[0] or [],
                    ),
                    main_loop,
                )
                result = future.result(timeout=30)
                if result.get("success"):
                    thread_chat_id = str(delivery_state.get("chat_id") or "")
                    thread_source = delivery_state.get("thread_source")
                    if thread_chat_id:
                        os.environ["HERMES_SESSION_CHAT_ID"] = thread_chat_id
                        os.environ["HERMES_SESSION_CHAT_TYPE"] = "thread"
                        os.environ["HERMES_SESSION_THREAD_ID"] = thread_chat_id
                    if thread_source and getattr(thread_source, "chat_name", None):
                        os.environ["HERMES_SESSION_CHAT_NAME"] = str(thread_source.chat_name)
                    thread_session_key = str(delivery_state.get("thread_session_key") or "").strip()
                    if thread_session_key:
                        os.environ["HERMES_SESSION_KEY"] = thread_session_key
                return result

            agent = AIAgent(
                model=model,
                api_key=api_key,
                base_url=base_url,
                max_iterations=max_iterations,
                quiet_mode=True,
                verbose_logging=False,
                enabled_toolsets=enabled_toolsets,
                ephemeral_system_prompt=combined_ephemeral or None,
                prefill_messages=self._prefill_messages or None,
                reasoning_config=self._reasoning_config,
                model_extra_body=model_extra_body,
                session_id=session_id,
                tool_progress_callback=progress_callback if tool_progress_enabled else None,
                fork_thread_callback=fork_thread_callback,
                platform=platform_key,
                honcho_session_key=session_key,
                session_db=self._session_db,
                session_db_writes=False,
                context_cwd=self._context_cwd,
            )
            
            # Store agent reference for interrupt support
            agent_holder[0] = agent
            # Capture the full tool definitions for transcript logging
            tools_holder[0] = agent.tools if hasattr(agent, 'tools') else None
            
            # Convert history to agent format.
            # Two cases:
            #   1. Normal path (from transcript): simple {role, content, timestamp} dicts
            #      - Strip timestamps, keep role+content
            #   2. Interrupt path (from agent result["messages"]): full agent messages
            #      that may include tool_calls, tool_call_id, reasoning, etc.
            #      - These must be passed through intact so the API sees valid
            #        assistant→tool sequences (dropping tool_calls causes 500 errors)
            agent_history = []
            last_signature = None

            def _signature(msg_obj: Dict[str, Any]) -> Optional[tuple]:
                """Stable signature for lightweight transcript de-duplication."""
                try:
                    payload = json.dumps(msg_obj, sort_keys=True, ensure_ascii=False)
                except Exception:
                    payload = str(msg_obj)
                return (msg_obj.get("role"), payload)

            def _append_history(msg_obj: Dict[str, Any]) -> None:
                nonlocal last_signature
                # Older sessions may contain duplicate assistant/tool rows from
                # legacy double-write paths. Skip only exact adjacent duplicates.
                role = msg_obj.get("role")
                sig = _signature(msg_obj)
                if role in ("assistant", "tool", "function") and sig == last_signature:
                    return
                agent_history.append(msg_obj)
                last_signature = sig

            for msg in history:
                role = msg.get("role")
                if not role:
                    continue
                
                # Skip metadata entries (tool definitions, session info)
                # -- these are for transcript logging, not for the LLM
                if role in ("session_meta",):
                    continue
                
                # Skip system messages -- the agent rebuilds its own system prompt
                if role == "system":
                    continue
                
                # Rich agent messages (tool_calls, tool results) must be passed
                # through intact so the API sees valid assistant→tool sequences
                has_tool_calls = "tool_calls" in msg
                has_tool_call_id = "tool_call_id" in msg
                is_tool_message = role == "tool"
                
                if has_tool_calls or has_tool_call_id or is_tool_message:
                    clean_msg = {k: v for k, v in msg.items() if k != "timestamp"}
                    if "content" in clean_msg:
                        clean_msg["content"] = self._sanitize_multimodal_history_content(clean_msg.get("content"))
                    _append_history(clean_msg)
                else:
                    # Simple text message - just need role and content
                    content = msg.get("content")
                    if content:
                        content = self._sanitize_multimodal_history_content(content)
                        # Tag cross-platform mirror messages so the agent knows their origin
                        if msg.get("mirror"):
                            mirror_src = msg.get("mirror_source", "another session")
                            content = f"[Delivered from {mirror_src}] {content}"
                        _append_history({"role": role, "content": content})
            
            history_input_len = len(agent_history)
            user_payload = self._sanitize_multimodal_history_content(message)
            sandbox_task_id = self._sandbox_task_id_for_session(
                session_key=session_key,
                source=source,
                session_id=session_id,
            )
            result = agent.run_conversation(
                user_payload,
                conversation_history=agent_history,
                task_id=sandbox_task_id,
            )
            result_holder[0] = result
            
            # Return final response, or a message if something went wrong
            final_response = result.get("final_response")
            if not final_response:
                error_msg = f"⚠️ {result['error']}" if result.get("error") else "(No response generated)"
                return {
                    "final_response": error_msg,
                    "messages": result.get("messages", []),
                    "api_calls": result.get("api_calls", 0),
                    "tools": tools_holder[0] or [],
                    "history_input_len": history_input_len,
                    "request_usage": result.get("request_usage", {}),
                }
            
            # Scan tool results for MEDIA:<path> tags that need to be delivered
            # as native audio/file attachments.  The TTS tool embeds MEDIA: tags
            # in its JSON response, but the model's final text reply usually
            # doesn't include them.  We collect unique tags from tool results and
            # append any that aren't already present in the final response, so the
            # adapter's extract_media() can find and deliver the files exactly once.
            if "MEDIA:" not in final_response:
                media_tags = []
                has_voice_directive = False
                for msg in result.get("messages", []):
                    if msg.get("role") == "tool" or msg.get("role") == "function":
                        content = msg.get("content", "")
                        if "MEDIA:" in content:
                            for match in re.finditer(r'MEDIA:(\S+)', content):
                                path = match.group(1).strip().rstrip('",}')
                                if path:
                                    media_tags.append(f"MEDIA:{path}")
                            if "[[audio_as_voice]]" in content:
                                has_voice_directive = True
                
                if media_tags:
                    # Deduplicate while preserving order
                    seen = set()
                    unique_tags = []
                    for tag in media_tags:
                        if tag not in seen:
                            seen.add(tag)
                            unique_tags.append(tag)
                    if has_voice_directive:
                        unique_tags.insert(0, "[[audio_as_voice]]")
                    final_response = final_response + "\n" + "\n".join(unique_tags)
            
            return {
                "final_response": final_response,
                "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
                "api_calls": result_holder[0].get("api_calls", 0) if result_holder[0] else 0,
                "tools": tools_holder[0] or [],
                "history_input_len": history_input_len,
                "request_usage": result_holder[0].get("request_usage", {}) if result_holder[0] else {},
            }
        
        # Start progress message sender if enabled
        progress_task = None
        if tool_progress_enabled:
            progress_task = asyncio.create_task(send_progress_messages())
        
        # Track this agent as running for this session (for interrupt support)
        # We do this in a callback after the agent is created
        async def track_agent():
            # Wait for agent to be created
            while agent_holder[0] is None:
                await asyncio.sleep(0.05)
            if session_key:
                self._running_agents[session_key] = agent_holder[0]
        
        tracking_task = asyncio.create_task(track_agent())
        
        # Monitor for interrupts from the adapter (new messages arriving)
        async def monitor_for_interrupt():
            adapter = self.adapters.get(source.platform)
            if not adapter:
                return

            while True:
                await asyncio.sleep(0.2)  # Check every 200ms
                chat_ids = [source.chat_id]
                current_chat_id = str(delivery_state.get("chat_id") or "").strip()
                if current_chat_id and current_chat_id not in chat_ids:
                    chat_ids.append(current_chat_id)

                for chat_id in chat_ids:
                    if hasattr(adapter, 'has_pending_interrupt') and adapter.has_pending_interrupt(chat_id):
                        agent = agent_holder[0]
                        if agent:
                            pending_event = adapter.get_pending_message(chat_id)
                            pending_text = pending_event.text if pending_event else None
                            logger.debug("Interrupt detected from adapter, signaling agent...")
                            agent.interrupt(pending_text)
                            break
                else:
                    continue
                break
        
        interrupt_monitor = asyncio.create_task(monitor_for_interrupt())
        
        try:
            # Run in thread pool to not block
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, run_sync)
            
            # Check if we were interrupted and have a pending message
            result = result_holder[0]
            adapter = self.adapters.get(source.platform)
            
            # Get pending message from adapter if interrupted
            pending = None
            pending_event = None
            if result and result.get("interrupted") and adapter:
                pending_chat_ids = [source.chat_id]
                current_chat_id = str(delivery_state.get("chat_id") or "").strip()
                if current_chat_id and current_chat_id not in pending_chat_ids:
                    pending_chat_ids.append(current_chat_id)

                for chat_id in pending_chat_ids:
                    pending_event = adapter.get_pending_message(chat_id)
                    if pending_event:
                        break
                if pending_event:
                    pending = pending_event.text
                elif result.get("interrupt_message"):
                    pending = result.get("interrupt_message")
            
            if pending:
                logger.debug("Processing interrupted message: '%s...'", pending[:40])
                
                # Clear the adapter's interrupt event so the next _run_agent call
                # doesn't immediately re-trigger the interrupt before the new agent
                # even makes its first API call (this was causing an infinite loop).
                if adapter and hasattr(adapter, "_active_sessions"):
                    for chat_id in pending_chat_ids:
                        if chat_id in adapter._active_sessions:
                            adapter._active_sessions[chat_id].clear()
                
                # Don't send the interrupted response to the user — it's just noise
                # like "Operation interrupted." They already know they sent a new
                # message, so go straight to processing it.
                
                # Now process the pending message with updated history
                pending_source = pending_event.source if pending_event and pending_event.source else source
                thread_source = delivery_state.get("thread_source") if isinstance(delivery_state, dict) else None
                thread_session_id = str(
                    delivery_state.get("thread_session_id") or ""
                ).strip() if isinstance(delivery_state, dict) else ""
                thread_chat_id = str(getattr(thread_source, "chat_id", "") or "").strip()
                pending_chat_id = str(getattr(pending_source, "chat_id", "") or "").strip()
                source_chat_id = str(getattr(source, "chat_id", "") or "").strip()
                crossed_fork_boundary = bool(
                    pending_event
                    and thread_session_id
                    and thread_chat_id
                    and pending_chat_id == thread_chat_id
                    and pending_chat_id != source_chat_id
                )
                pending_store = getattr(adapter, "_pending_messages", None) if adapter else None
                if crossed_fork_boundary and isinstance(pending_store, dict):
                    pending_store[pending_chat_id] = pending_event
                    deferred_response = dict(response or {})
                    deferred_response["final_response"] = ""
                    deferred_response["deferred_pending_event"] = True
                    deferred_response["interrupted_handoff"] = True
                    return deferred_response

                speaker_changed = bool(pending_event) and not self._same_speaker(source, pending_source)
                updated_history = history if speaker_changed else result.get("messages", history)
                resumed_event = pending_event or event
                resumed_source = pending_source if pending_event else source
                resumed_delivery_state = delivery_state if isinstance(delivery_state, dict) else {}
                resumed_delivery_state.clear()
                resumed_delivery_state.update(
                    self._new_delivery_state(resumed_source, resumed_event)
                )
                resumed_message = pending
                if pending_event:
                    interrupt_note = (
                        self._discord_interrupt_note(source, pending_source)
                        if speaker_changed else ""
                    )
                    resumed_message, _ = await self._build_event_message_payload(
                        pending_event,
                        pending_source,
                        interrupt_note=interrupt_note,
                    )
                return await self._run_agent(
                    message=resumed_message,
                    context_prompt=context_prompt,
                    history=updated_history,
                    source=resumed_source,
                    session_id=session_id,
                    session_key=session_key,
                    event=resumed_event,
                    delivery_state=resumed_delivery_state,
                )
        finally:
            # Stop progress sender and interrupt monitor
            if progress_task:
                progress_task.cancel()
            interrupt_monitor.cancel()
            
            # Clean up tracking
            tracking_task.cancel()
            if session_key and session_key in self._running_agents:
                del self._running_agents[session_key]
            
            # Wait for cancelled tasks
            for task in [progress_task, interrupt_monitor, tracking_task]:
                if task:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
        return response


def _start_cron_ticker(stop_event: threading.Event, adapters=None, interval: int = 60):
    """
    Background thread that ticks the cron scheduler at a regular interval.
    
    Runs inside the gateway process so cronjobs fire automatically without
    needing a separate `hermes cron daemon` or system cron entry.

    Also refreshes the channel directory every 5 minutes and prunes the
    image/audio/document cache once per hour.
    """
    from cron.scheduler import tick as cron_tick
    from gateway.platforms.base import cleanup_image_cache, cleanup_document_cache

    IMAGE_CACHE_EVERY = 60   # ticks — once per hour at default 60s interval
    CHANNEL_DIR_EVERY = 5    # ticks — every 5 minutes

    logger.info("Cron ticker started (interval=%ds)", interval)
    tick_count = 0
    while not stop_event.is_set():
        try:
            cron_tick(verbose=False)
        except Exception as e:
            logger.debug("Cron tick error: %s", e)

        tick_count += 1

        if tick_count % CHANNEL_DIR_EVERY == 0 and adapters:
            try:
                from gateway.channel_directory import build_channel_directory
                build_channel_directory(adapters)
            except Exception as e:
                logger.debug("Channel directory refresh error: %s", e)

        if tick_count % IMAGE_CACHE_EVERY == 0:
            try:
                removed = cleanup_image_cache(max_age_hours=24)
                if removed:
                    logger.info("Image cache cleanup: removed %d stale file(s)", removed)
            except Exception as e:
                logger.debug("Image cache cleanup error: %s", e)
            try:
                removed = cleanup_document_cache(max_age_hours=24)
                if removed:
                    logger.info("Document cache cleanup: removed %d stale file(s)", removed)
            except Exception as e:
                logger.debug("Document cache cleanup error: %s", e)

        stop_event.wait(timeout=interval)
    logger.info("Cron ticker stopped")


async def start_gateway(config: Optional[GatewayConfig] = None) -> bool:
    """
    Start the gateway and run until interrupted.
    
    This is the main entry point for running the gateway.
    Returns True if the gateway ran successfully, False if it failed to start.
    A False return causes a non-zero exit code so systemd can auto-restart.
    """
    # Configure rotating file log so gateway output is persisted for debugging
    log_dir = _hermes_home / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / 'gateway.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)

    runner = GatewayRunner(config)
    
    # Set up signal handlers
    def signal_handler():
        asyncio.create_task(runner.stop())
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass
    
    # Start the gateway
    success = await runner.start()
    if not success:
        return False
    
    # Write PID file so CLI can detect gateway is running
    import atexit
    from gateway.status import write_pid_file, remove_pid_file
    write_pid_file()
    atexit.register(remove_pid_file)
    
    # Start background cron ticker so scheduled jobs fire automatically
    cron_stop = threading.Event()
    cron_thread = threading.Thread(
        target=_start_cron_ticker,
        args=(cron_stop,),
        kwargs={"adapters": runner.adapters},
        daemon=True,
        name="cron-ticker",
    )
    cron_thread.start()
    
    # Wait for shutdown
    await runner.wait_for_shutdown()
    
    # Stop cron ticker cleanly
    cron_stop.set()
    cron_thread.join(timeout=5)
    
    return True


def main():
    """CLI entry point for the gateway."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hermes Gateway - Multi-platform messaging")
    parser.add_argument("--config", "-c", help="Path to gateway config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config = None
    if args.config:
        import json
        with open(args.config) as f:
            data = json.load(f)
            config = GatewayConfig.from_dict(data)
    
    # Run the gateway - exit with code 1 if no platforms connected,
    # so systemd Restart=on-failure will retry on transient errors (e.g. DNS)
    success = asyncio.run(start_gateway(config))
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
