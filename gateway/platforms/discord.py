"""
Discord platform adapter.

Uses discord.py library for:
- Receiving messages from servers and DMs
- Sending responses back
- Handling threads and channels
"""

import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import discord
    from discord import Message as DiscordMessage, Intents
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None
    DiscordMessage = Any
    Intents = Any
    commands = None

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.discord_archive import DiscordArchiveDB, default_archive_db_path
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_url,
    cache_audio_from_url,
)


def check_discord_requirements() -> bool:
    """Check if Discord dependencies are available."""
    return DISCORD_AVAILABLE


class DiscordAdapter(BasePlatformAdapter):
    """
    Discord bot adapter.
    
    Handles:
    - Receiving messages from servers and DMs
    - Sending responses with Discord markdown
    - Thread support
    - Native slash commands (/ask, /reset, /status, /stop)
    - Button-based exec approvals
    - Auto-threading for long conversations
    - Reaction-based feedback
    """
    
    # Discord message limits
    MAX_MESSAGE_LENGTH = 2000
    CUSTOM_EMOJI_RE = re.compile(r"<(?P<animated>a?):(?P<name>[A-Za-z0-9_]{1,64}):(?P<id>\d+)>")
    USER_MENTION_RE = re.compile(r"<@!?(?P<id>\d+)>")
    
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DISCORD)
        self._client: Optional[commands.Bot] = None
        self._ready_event = asyncio.Event()
        self._allowed_user_ids: set = set()  # For button approval authorization
        self._archive_db: Optional[DiscordArchiveDB] = None
        self._bootstrapped_channels: Set[str] = set()
        self._context_header_sent_channels: Set[str] = set()
        self._force_fresh_context_channels: Set[str] = set()
        self._archive_worker_stop = asyncio.Event()
        self._frontfill_task: Optional[asyncio.Task] = None
        self._backfill_task: Optional[asyncio.Task] = None
        self._active_history_channels: Set[str] = set()
        self._frontfill_rr_index = 0
        self._backfill_rr_index = 0
    
    async def connect(self) -> bool:
        """Connect to Discord and start receiving events."""
        if not DISCORD_AVAILABLE:
            print(f"[{self.name}] discord.py not installed. Run: pip install discord.py")
            return False
        
        if not self.config.token:
            print(f"[{self.name}] No bot token configured")
            return False
        
        try:
            if self._archive_enabled():
                self._archive_db = DiscordArchiveDB(self._archive_db_path())

            # Set up intents -- members intent needed for username-to-ID resolution
            intents = Intents.default()
            intents.message_content = True
            intents.dm_messages = self._dms_enabled()
            intents.guild_messages = True
            intents.members = True
            
            # Create bot
            self._client = commands.Bot(
                command_prefix="!",  # Not really used, we handle raw messages
                intents=intents,
            )
            
            # Parse allowed user entries (may contain usernames or IDs)
            allowed_env = os.getenv("DISCORD_ALLOWED_USERS", "")
            if allowed_env:
                self._allowed_user_ids = {
                    uid.strip() for uid in allowed_env.split(",") if uid.strip()
                }
            
            adapter_self = self  # capture for closure
            
            # Register event handlers
            @self._client.event
            async def on_ready():
                print(f"[{adapter_self.name}] Connected as {adapter_self._client.user}")
                
                # Resolve any usernames in the allowed list to numeric IDs
                await adapter_self._resolve_allowed_usernames()
                
                # Sync slash commands with Discord
                try:
                    synced = await adapter_self._client.tree.sync()
                    print(f"[{adapter_self.name}] Synced {len(synced)} slash command(s)")
                except Exception as e:
                    print(f"[{adapter_self.name}] Slash command sync failed: {e}")
                try:
                    await adapter_self._start_archive_workers()
                except Exception as e:
                    logger.debug("Discord archive worker startup failed: %s", e)
                adapter_self._ready_event.set()
            
            @self._client.event
            async def on_message(message: DiscordMessage):
                # Ignore bot's own messages
                if message.author == self._client.user:
                    return
                await self._handle_message(message)

            @self._client.event
            async def on_message_edit(before: DiscordMessage, after: DiscordMessage):
                # Ignore bot's own edits
                if after.author == self._client.user:
                    return
                await self._archive_message_edit(before, after)

            @self._client.event
            async def on_raw_message_delete(payload: discord.RawMessageDeleteEvent):
                if not adapter_self._archive_db:
                    return
                try:
                    adapter_self._archive_db.mark_deleted(
                        message_id=str(payload.message_id),
                        channel_id=str(payload.channel_id) if payload.channel_id else None,
                        guild_id=str(payload.guild_id) if payload.guild_id else None,
                    )
                except Exception as e:
                    logger.debug("Discord archive delete mark failed: %s", e)

            @self._client.event
            async def on_raw_bulk_message_delete(payload: discord.RawBulkMessageDeleteEvent):
                if not adapter_self._archive_db:
                    return
                guild_id = str(payload.guild_id) if payload.guild_id else None
                channel_id = str(payload.channel_id) if payload.channel_id else None
                for mid in payload.message_ids:
                    try:
                        adapter_self._archive_db.mark_deleted(
                            message_id=str(mid),
                            channel_id=channel_id,
                            guild_id=guild_id,
                        )
                    except Exception:
                        pass
            
            # Register slash commands
            self._register_slash_commands()
            
            # Start the bot in background
            asyncio.create_task(self._client.start(self.config.token))
            
            # Wait for ready
            await asyncio.wait_for(self._ready_event.wait(), timeout=30)
            
            self._running = True
            return True
            
        except asyncio.TimeoutError:
            print(f"[{self.name}] Timeout waiting for connection")
            return False
        except Exception as e:
            print(f"[{self.name}] Failed to connect: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Discord."""
        await self._stop_archive_workers()
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                print(f"[{self.name}] Error during disconnect: {e}")
        
        self._running = False
        self._client = None
        self._ready_event.clear()
        self._bootstrapped_channels.clear()
        self._context_header_sent_channels.clear()
        self._force_fresh_context_channels.clear()
        self._active_history_channels.clear()
        self._frontfill_rr_index = 0
        self._backfill_rr_index = 0
        if self._archive_db:
            try:
                self._archive_db.close()
            except Exception:
                pass
            self._archive_db = None
        print(f"[{self.name}] Disconnected")
    
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Send a message to a Discord channel."""
        if not self._client:
            return SendResult(success=False, error="Not connected")
        
        try:
            # Get the channel
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            if isinstance(channel, discord.DMChannel) and not self._dms_enabled():
                return SendResult(
                    success=False,
                    error="Discord DMs are disabled (DISCORD_ENABLE_DMS=false)",
                )
            
            # Format and split message if needed
            formatted = self.format_message(content)
            chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
            
            message_ids = []
            reference = None
            
            if reply_to:
                try:
                    ref_msg = await channel.fetch_message(int(reply_to))
                    reference = ref_msg
                except Exception as e:
                    logger.debug("Could not fetch reply-to message: %s", e)
            
            for i, chunk in enumerate(chunks):
                msg = await channel.send(
                    content=chunk,
                    reference=reference if i == 0 else None,
                )
                message_ids.append(str(msg.id))
            
            return SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={"message_ids": message_ids}
            )
            
        except Exception as e:
            return SendResult(success=False, error=str(e))
    
    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send audio as a Discord file attachment."""
        if not self._client:
            return SendResult(success=False, error="Not connected")
        
        try:
            import io
            
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            
            if not os.path.exists(audio_path):
                return SendResult(success=False, error=f"Audio file not found: {audio_path}")
            
            # Determine filename from path
            filename = os.path.basename(audio_path)
            
            with open(audio_path, "rb") as f:
                file = discord.File(io.BytesIO(f.read()), filename=filename)
                msg = await channel.send(
                    content=caption if caption else None,
                    file=file,
                )
                return SendResult(success=True, message_id=str(msg.id))
        
        except Exception as e:
            print(f"[{self.name}] Failed to send audio: {e}")
            return await super().send_voice(chat_id, audio_path, caption, reply_to)
    
    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send an image natively as a Discord file attachment."""
        if not self._client:
            return SendResult(success=False, error="Not connected")
        
        try:
            import aiohttp
            
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            
            # Download the image and send as a Discord file attachment
            # (Discord renders attachments inline, unlike plain URLs)
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to download image: HTTP {resp.status}")
                    
                    image_data = await resp.read()
                    
                    # Determine filename from URL or content type
                    content_type = resp.headers.get("content-type", "image/png")
                    ext = "png"
                    if "jpeg" in content_type or "jpg" in content_type:
                        ext = "jpg"
                    elif "gif" in content_type:
                        ext = "gif"
                    elif "webp" in content_type:
                        ext = "webp"
                    
                    import io
                    file = discord.File(io.BytesIO(image_data), filename=f"image.{ext}")
                    
                    msg = await channel.send(
                        content=caption if caption else None,
                        file=file,
                    )
                    return SendResult(success=True, message_id=str(msg.id))
        
        except ImportError:
            print(f"[{self.name}] aiohttp not installed, falling back to URL. Run: pip install aiohttp")
            return await super().send_image(chat_id, image_url, caption, reply_to)
        except Exception as e:
            print(f"[{self.name}] Failed to send image attachment, falling back to URL: {e}")
            return await super().send_image(chat_id, image_url, caption, reply_to)
    
    async def send_typing(self, chat_id: str) -> None:
        """Send typing indicator."""
        if self._client:
            try:
                channel = self._client.get_channel(int(chat_id))
                if channel:
                    await channel.typing()
            except Exception:
                pass  # Ignore typing indicator failures
    
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Discord channel."""
        if not self._client:
            return {"name": "Unknown", "type": "dm"}
        
        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            
            if not channel:
                return {"name": str(chat_id), "type": "dm"}
            
            # Determine channel type
            if isinstance(channel, discord.DMChannel):
                chat_type = "dm"
                name = channel.recipient.name if channel.recipient else str(chat_id)
            elif isinstance(channel, discord.Thread):
                chat_type = "thread"
                name = channel.name
            elif isinstance(channel, discord.TextChannel):
                chat_type = "channel"
                name = f"#{channel.name}"
                if channel.guild:
                    name = f"{channel.guild.name} / {name}"
            else:
                chat_type = "channel"
                name = getattr(channel, "name", str(chat_id))
            
            return {
                "name": name,
                "type": chat_type,
                "guild_id": str(channel.guild.id) if hasattr(channel, "guild") and channel.guild else None,
                "guild_name": channel.guild.name if hasattr(channel, "guild") and channel.guild else None,
            }
        except Exception as e:
            return {"name": str(chat_id), "type": "dm", "error": str(e)}

    @staticmethod
    def _parse_int(value: Any, default: int) -> int:
        """Parse an integer value safely."""
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_truthy(value: Any, default: bool = False) -> bool:
        """Parse flexible boolean values."""
        if value is None:
            return default
        text = str(value).strip().strip("\"'").lower()
        if not text:
            return default
        if text in ("true", "1", "yes", "y", "on"):
            return True
        if text in ("false", "0", "no", "n", "off"):
            return False
        return default

    def _extra(self) -> Dict[str, Any]:
        return self.config.extra if isinstance(self.config.extra, dict) else {}

    @staticmethod
    def _parse_id_set(value: Any) -> Set[str]:
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            entries = [str(v).strip() for v in value]
        else:
            entries = [part.strip() for part in str(value).split(",")]
        return {entry for entry in entries if entry}

    def _allowed_channel_ids(self) -> Set[str]:
        raw_env = os.getenv("DISCORD_ALLOWED_CHANNEL_IDS", "")
        if raw_env.strip():
            return self._parse_id_set(raw_env)
        return self._parse_id_set(self._extra().get("allowed_channel_ids"))

    def _allowed_guild_ids(self) -> Set[str]:
        raw_env = os.getenv("DISCORD_ALLOWED_GUILD_IDS", "")
        if raw_env.strip():
            return self._parse_id_set(raw_env)
        return self._parse_id_set(self._extra().get("allowed_guild_ids"))

    def _dms_enabled(self) -> bool:
        raw_env = (os.getenv("DISCORD_ENABLE_DMS", "") or "").strip()
        if raw_env:
            return self._is_truthy(raw_env, default=True)
        return self._is_truthy(self._extra().get("enable_dms"), default=True)

    def _archive_enabled(self) -> bool:
        raw_env = os.getenv("DISCORD_ARCHIVE_ENABLED", "")
        if raw_env.strip():
            return self._is_truthy(raw_env, default=True)
        return self._is_truthy(self._extra().get("archive_enabled"), default=True)

    def _archive_db_path(self) -> str:
        raw_env = (os.getenv("DISCORD_ARCHIVE_DB_PATH", "") or "").strip()
        if raw_env:
            return os.path.expanduser(raw_env)
        raw_cfg = str(self._extra().get("archive_db_path", "") or "").strip()
        if raw_cfg:
            return os.path.expanduser(raw_cfg)
        return str(default_archive_db_path())

    def _fresh_context_limit(self) -> int:
        """
        Last-N context size for fresh channel turns.

        Used when:
        - no prior channel turn anchor exists
        - channel delta exceeds the reset threshold
        """
        raw_env = (os.getenv("DISCORD_FRESH_CONTEXT_LIMIT", "") or "").strip()
        if raw_env:
            value = self._parse_int(raw_env, 20)
        else:
            value = self._parse_int(self._extra().get("fresh_context_limit", 20), 20)
        return max(1, min(value, 200))

    def _delta_reset_threshold(self) -> int:
        """Auto-reset if non-bot message delta exceeds this threshold."""
        raw_env = (os.getenv("DISCORD_DELTA_RESET_THRESHOLD", "") or "").strip()
        if raw_env:
            value = self._parse_int(raw_env, 50)
        else:
            value = self._parse_int(self._extra().get("delta_reset_threshold", 50), 50)
        return max(1, min(value, 500))

    def _context_max_chars(self) -> int:
        """
        Max characters for the injected channel context block.

        Priority:
        1) DISCORD_CONTEXT_MAX_CHARS env var
        2) config.yaml -> gateway.discord.context_max_chars
        3) default (14000)
        """
        raw_env = (os.getenv("DISCORD_CONTEXT_MAX_CHARS", "") or "").strip()
        if raw_env:
            value = self._parse_int(raw_env, 14000)
        else:
            value = self._parse_int(self._extra().get("context_max_chars", 14000), 14000)
        return max(2000, min(value, 100000))

    def _empty_delta_fallback_enabled(self) -> bool:
        """
        If enabled, empty delta follow-ups fall back to fresh last-N context.

        Default is False so follow-up turns are delta-only.
        """
        raw_env = (os.getenv("DISCORD_CONTEXT_EMPTY_DELTA_FALLBACK", "") or "").strip()
        if raw_env:
            return self._is_truthy(raw_env, default=False)
        return self._is_truthy(self._extra().get("context_empty_delta_fallback"), default=False)

    def _full_scrape_enabled(self) -> bool:
        """
        Enable async archive workers:
        - front-filler: forward sync from the latest cursor
        - backfiller: backward sync to fill older channel history
        """
        raw_env = (os.getenv("DISCORD_FULL_SCRAPE_ENABLED", "") or "").strip()
        if raw_env:
            return self._is_truthy(raw_env, default=True)
        return self._is_truthy(self._extra().get("full_scrape_enabled"), default=True)

    def _full_scrape_interval_sec(self) -> int:
        # DCE-style defaults: continuous polling cadence, single-channel processing,
        # and deep per-channel pagination (100 message pages).
        raw_env = (os.getenv("DISCORD_FULL_SCRAPE_INTERVAL_SEC", "") or "").strip()
        if raw_env:
            value = self._parse_int(raw_env, 15)
        else:
            value = self._parse_int(self._extra().get("full_scrape_interval_sec", 15), 15)
        return max(15, min(value, 3600))

    def _full_scrape_max_channels_per_tick(self) -> int:
        raw_env = (os.getenv("DISCORD_FULL_SCRAPE_MAX_CHANNELS_PER_TICK", "") or "").strip()
        if raw_env:
            value = self._parse_int(raw_env, 1)
        else:
            value = self._parse_int(self._extra().get("full_scrape_max_channels_per_tick", 1), 1)
        return max(1, min(value, 100))

    def _full_scrape_max_pages_per_channel(self) -> int:
        raw_env = (os.getenv("DISCORD_FULL_SCRAPE_MAX_PAGES_PER_CHANNEL", "") or "").strip()
        if raw_env:
            value = self._parse_int(raw_env, 100)
        else:
            value = self._parse_int(self._extra().get("full_scrape_max_pages_per_channel", 100), 100)
        return max(1, min(value, 100))

    def _full_scrape_seed_limit(self) -> int:
        raw_env = (os.getenv("DISCORD_FULL_SCRAPE_SEED_LIMIT", "") or "").strip()
        if raw_env:
            value = self._parse_int(raw_env, 100)
        else:
            value = self._parse_int(self._extra().get("full_scrape_seed_limit", 100), 100)
        return max(20, min(value, 2000))

    def _full_scrape_include_threads(self) -> bool:
        raw_env = (os.getenv("DISCORD_FULL_SCRAPE_INCLUDE_THREADS", "") or "").strip()
        if raw_env:
            return self._is_truthy(raw_env, default=False)
        return self._is_truthy(self._extra().get("full_scrape_include_threads"), default=False)

    def _backfill_enabled(self) -> bool:
        raw_env = (os.getenv("DISCORD_BACKFILL_ENABLED", "") or "").strip()
        if raw_env:
            return self._is_truthy(raw_env, default=True)
        return self._is_truthy(self._extra().get("backfill_enabled"), default=True)

    def _backfill_interval_sec(self) -> int:
        raw_env = (os.getenv("DISCORD_BACKFILL_INTERVAL_SEC", "") or "").strip()
        if raw_env:
            value = self._parse_int(raw_env, 30)
        else:
            value = self._parse_int(self._extra().get("backfill_interval_sec", 30), 30)
        return max(30, min(value, 3600))

    def _backfill_max_channels_per_tick(self) -> int:
        raw_env = (os.getenv("DISCORD_BACKFILL_MAX_CHANNELS_PER_TICK", "") or "").strip()
        if raw_env:
            value = self._parse_int(raw_env, 1)
        else:
            value = self._parse_int(self._extra().get("backfill_max_channels_per_tick", 1), 1)
        return max(1, min(value, 50))

    def _backfill_max_pages_per_channel(self) -> int:
        raw_env = (os.getenv("DISCORD_BACKFILL_MAX_PAGES_PER_CHANNEL", "") or "").strip()
        if raw_env:
            value = self._parse_int(raw_env, 50)
        else:
            value = self._parse_int(self._extra().get("backfill_max_pages_per_channel", 50), 50)
        return max(1, min(value, 50))

    def _scrape_progress_every_pages(self) -> int:
        raw_env = (os.getenv("DISCORD_SCRAPE_PROGRESS_EVERY_PAGES", "") or "").strip()
        if raw_env:
            value = self._parse_int(raw_env, 3)
        else:
            value = self._parse_int(self._extra().get("scrape_progress_every_pages", 3), 3)
        return max(1, min(value, 100))

    def _can_view_channel(self, channel: Any) -> bool:
        """True when the bot has visibility in the given Discord channel."""
        if not DISCORD_AVAILABLE:
            return True
        if channel is None:
            return False
        if isinstance(channel, discord.DMChannel):
            return True

        guild = getattr(channel, "guild", None)
        if guild is None:
            return True

        member = getattr(guild, "me", None)
        if member is None:
            user = getattr(self._client, "user", None) if self._client else None
            user_id = getattr(user, "id", None)
            getter = getattr(guild, "get_member", None)
            if user_id is not None and callable(getter):
                try:
                    member = getter(int(user_id))
                except Exception:
                    member = None
        if member is None:
            return False

        permission_resolver = getattr(channel, "permissions_for", None)
        if not callable(permission_resolver):
            return False

        try:
            perms = permission_resolver(member)
        except Exception:
            return False
        return bool(getattr(perms, "view_channel", False))

    def _is_channel_allowed(self, channel: Any) -> bool:
        """Check guild/channel allowlists, with optional DM blocking."""
        if not DISCORD_AVAILABLE:
            return True
        if not self._can_view_channel(channel):
            return False
        if isinstance(channel, discord.DMChannel):
            return self._dms_enabled()

        allowed_channels = self._allowed_channel_ids()
        channel_id = str(getattr(channel, "id", ""))
        if allowed_channels and channel_id not in allowed_channels:
            return False

        allowed_guilds = self._allowed_guild_ids()
        guild_id = ""
        guild = getattr(channel, "guild", None)
        if guild is not None and getattr(guild, "id", None) is not None:
            guild_id = str(guild.id)

        if allowed_guilds:
            if not guild_id:
                return False
            if guild_id not in allowed_guilds:
                return False

        return True

    @staticmethod
    def _is_gif_attachment(content_type: str, filename: str = "") -> bool:
        """Return True when an attachment is a GIF image."""
        ctype = (content_type or "").lower()
        if ctype.startswith("image/") and "gif" in ctype:
            return True
        return str(filename or "").lower().endswith(".gif")

    def _extract_static_custom_emojis(self, text: str) -> List[Tuple[str, str]]:
        """
        Parse static custom emoji tags from Discord message content.

        Returns (emoji_name, image_url) pairs. Animated emojis are ignored.
        URLs request 256px PNGs for compact multimodal input.
        """
        if not text:
            return []
        seen_ids: Set[str] = set()
        rows: List[Tuple[str, str]] = []
        for match in self.CUSTOM_EMOJI_RE.finditer(text):
            if match.group("animated"):
                continue
            emoji_id = match.group("id")
            if emoji_id in seen_ids:
                continue
            seen_ids.add(emoji_id)
            emoji_name = match.group("name") or f"emoji_{emoji_id}"
            emoji_url = (
                f"https://cdn.discordapp.com/emojis/{emoji_id}.png"
                f"?size=256&quality=lossless"
            )
            rows.append((emoji_name, emoji_url))
        return rows

    @staticmethod
    def _attachment_payload(att: Any) -> Dict[str, Any]:
        return {
            "id": str(getattr(att, "id", "")),
            "filename": getattr(att, "filename", ""),
            "size": getattr(att, "size", 0),
            "url": getattr(att, "url", ""),
            "content_type": getattr(att, "content_type", None),
        }

    @staticmethod
    def _is_non_gif_image_attachment(content_type: str, filename: str) -> bool:
        """Return True for image attachments we should send as multimodal input."""
        ctype = (content_type or "").strip().lower()
        name = (filename or "").strip().lower()
        if ctype.startswith("image/"):
            if ctype == "image/gif" or name.endswith(".gif"):
                return False
            return True
        return name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"))

    def _followup_image_lookback(self) -> int:
        """
        Optional override for image carryover scan depth.

        Default behavior follows the same largest window already used by
        channel-context collection (fresh last-N vs delta threshold).
        """
        raw_env = (os.getenv("DISCORD_FOLLOWUP_IMAGE_LOOKBACK", "") or "").strip()
        if raw_env:
            return max(10, min(self._parse_int(raw_env, 50), 500))

        raw_cfg = self._extra().get("followup_image_lookback")
        if raw_cfg not in (None, ""):
            return max(10, min(self._parse_int(raw_cfg, 50), 500))

        return max(self._fresh_context_limit(), self._delta_reset_threshold())

    def _followup_image_window_limit(self) -> int:
        """
        Max rows to scan for carryover images.

        Cap carryover scanning to the same largest bound used by context
        strategies (fresh window and delta threshold).
        """
        context_bound = max(self._fresh_context_limit(), self._delta_reset_threshold())
        return max(1, min(self._followup_image_lookback(), context_bound))

    def _followup_image_max_count(self) -> int:
        """Max number of carryover images to attach on text-only follow-ups."""
        raw_env = (os.getenv("DISCORD_FOLLOWUP_IMAGE_MAX_COUNT", "") or "").strip()
        if raw_env:
            value = self._parse_int(raw_env, 2)
        else:
            value = self._parse_int(self._extra().get("followup_image_max_count", 2), 2)
        return max(1, min(value, 6))

    def _followup_image_max_age_seconds(self) -> int:
        raw = (os.getenv("DISCORD_FOLLOWUP_IMAGE_MAX_AGE_SECONDS", "7200") or "").strip()
        try:
            return max(0, min(int(raw), 7 * 24 * 3600))
        except Exception:
            return 7200

    def _followup_image_candidate_rows(self, channel_id: str) -> List[Dict[str, Any]]:
        """
        Collect candidate rows for follow-up image carryover.

        Mirrors the same archive-window strategy as channel text context:
        - fresh last-N when no anchor (or forced reset condition)
        - delta-after-anchor with threshold limit otherwise
        """
        if not self._archive_db:
            return []

        channel_id = str(channel_id or "").strip()
        if not channel_id:
            return []

        fresh_limit = self._fresh_context_limit()
        threshold = self._delta_reset_threshold()
        anchor = self._archive_db.get_turn_anchor(channel_id)
        rows: List[Dict[str, Any]] = []

        if anchor:
            delta = self._archive_db.count_new_non_bot_messages(channel_id, anchor)
            if delta > threshold:
                rows = self._archive_db.list_recent_messages(
                    channel_id=channel_id,
                    limit=fresh_limit,
                    include_bots=False,
                )
            else:
                rows = self._archive_db.list_messages_after(
                    channel_id=channel_id,
                    after_message_id=anchor,
                    limit=threshold,
                    include_bots=False,
                )
                if not rows and self._empty_delta_fallback_enabled():
                    rows = self._archive_db.list_recent_messages(
                        channel_id=channel_id,
                        limit=fresh_limit,
                        include_bots=False,
                    )
        else:
            rows = self._archive_db.list_recent_messages(
                channel_id=channel_id,
                limit=fresh_limit,
                include_bots=False,
            )

        scan_limit = self._followup_image_window_limit()
        if len(rows) > scan_limit:
            rows = rows[-scan_limit:]
        return rows

    def _collect_recent_channel_image_items(
        self,
        message: DiscordMessage,
        max_images: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Return recent image attachments from the channel for follow-up turns.

        Used when the current turn has text but no new image upload.
        """
        if not self._archive_db:
            return []

        channel_id = str(getattr(message.channel, "id", ""))
        current_message_id = str(getattr(message, "id", ""))
        if not channel_id:
            return []

        rows = self._followup_image_candidate_rows(channel_id)
        if not rows:
            return []

        now_ts = (
            float(message.created_at.timestamp())
            if getattr(message, "created_at", None) is not None
            else float(datetime.now().timestamp())
        )
        max_age = self._followup_image_max_age_seconds()

        carryover: List[Dict[str, str]] = []
        seen_urls: Set[str] = set()
        for row in reversed(rows):  # newest -> oldest
            if str(row.get("message_id") or "") == current_message_id:
                continue

            ts = float(row.get("created_at") or 0)
            if max_age > 0 and ts > 0 and (now_ts - ts) > max_age:
                break

            for att in row.get("attachments", []) or []:
                if not isinstance(att, dict):
                    continue
                content_type = str(att.get("content_type") or "")
                filename = str(att.get("filename") or "")
                if not self._is_non_gif_image_attachment(content_type, filename):
                    continue
                url = str(att.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                carryover.append(
                    {
                        "path": url,
                        "media_type": content_type or "image/*",
                        "source_url": url,
                        "emoji_name": "",
                    }
                )
                if len(carryover) >= max_images:
                    return carryover

        return carryover

    def _collect_recent_user_image_items(
        self,
        message: DiscordMessage,
        max_images: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Backward-compatible wrapper for older call sites/tests.
        """
        return self._collect_recent_channel_image_items(message, max_images=max_images)

    def _replace_user_mentions(self, content: str, message: DiscordMessage) -> str:
        """
        Replace Discord user mention IDs with global usernames.

        Example:
          "<@123>" or "<@!123>" -> "@username"
        """
        text = content or ""
        mentions = list(getattr(message, "mentions", []) or [])
        if not text or not mentions:
            return text

        mention_names: Dict[str, str] = {}
        for user in mentions:
            user_id = str(getattr(user, "id", "")).strip()
            if not user_id:
                continue
            # Prefer global username over server-specific nickname.
            username = (getattr(user, "name", "") or "").strip()
            if username:
                mention_names[user_id] = username

        if not mention_names:
            return text

        def _sub(match: re.Match) -> str:
            user_id = match.group("id")
            username = mention_names.get(user_id)
            if not username:
                return match.group(0)
            return f"@{username}"

        return self.USER_MENTION_RE.sub(_sub, text)

    @staticmethod
    def _clean_text_block(value: Any) -> str:
        text = str(value or "")
        if not text:
            return ""
        # Normalize line endings while preserving user-visible formatting.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return text.strip()

    def _dedupe_chunk_key(self, text: str, message: DiscordMessage) -> str:
        """
        Build a stable key for chunk deduplication.

        Normalizes Discord mention syntax so equivalent forms like
        "<@123>" and "@username" collapse to one chunk.
        """
        cleaned = self._clean_text_block(text)
        if not cleaned:
            return ""
        mention_normalized = self._replace_user_mentions(cleaned, message)
        # Collapse all whitespace in the dedupe key (presentation keeps formatting).
        return " ".join(mention_normalized.split()).casefold()

    @classmethod
    def _embed_text(cls, embed: Any) -> str:
        if embed is None:
            return ""

        parts: List[str] = []
        for attr in ("title", "description", "url"):
            value = cls._clean_text_block(getattr(embed, attr, ""))
            if value:
                parts.append(value)

        author_obj = getattr(embed, "author", None)
        author_name = cls._clean_text_block(getattr(author_obj, "name", "")) if author_obj else ""
        if author_name:
            parts.append(author_name)

        for field in getattr(embed, "fields", []) or []:
            name = cls._clean_text_block(getattr(field, "name", ""))
            value = cls._clean_text_block(getattr(field, "value", ""))
            if name and value:
                parts.append(f"{name}: {value}")
            elif value:
                parts.append(value)

        footer_obj = getattr(embed, "footer", None)
        footer_text = cls._clean_text_block(getattr(footer_obj, "text", "")) if footer_obj else ""
        if footer_text:
            parts.append(footer_text)

        return "\n".join(parts).strip()

    def _materialize_message_text(self, message: DiscordMessage, base_content: Optional[str] = None) -> str:
        """
        Build a text representation for a Discord message.

        Discord forwarded messages can have empty message.content while carrying
        the actual payload inside message_snapshots, so fold those in as text.
        """
        chunks: List[str] = []

        main_content = base_content
        if main_content is None:
            main_content = getattr(message, "content", "") or ""
        main_content = self._replace_user_mentions(main_content, message)
        main_text = self._clean_text_block(main_content)
        if main_text:
            chunks.append(main_text)

        system_text = self._clean_text_block(
            self._replace_user_mentions(getattr(message, "system_content", "") or "", message)
        )
        if system_text:
            chunks.append(system_text)

        for embed in getattr(message, "embeds", []) or []:
            embed_text = self._embed_text(embed)
            if embed_text:
                chunks.append(embed_text)

        for snapshot in getattr(message, "message_snapshots", []) or []:
            snapshot_chunks: List[str] = []
            snapshot_content = self._clean_text_block(
                self._replace_user_mentions(getattr(snapshot, "content", "") or "", message)
            )
            if snapshot_content:
                snapshot_chunks.append(snapshot_content)
            for embed in getattr(snapshot, "embeds", []) or []:
                embed_text = self._embed_text(embed)
                if embed_text:
                    snapshot_chunks.append(embed_text)
            if snapshot_chunks:
                chunks.append("[forwarded message]\n" + "\n".join(snapshot_chunks))

        # Stable de-dup so we do not repeat equivalent content from overlapping fields.
        deduped: List[str] = []
        seen: Set[str] = set()
        for chunk in chunks:
            cleaned = self._clean_text_block(chunk)
            if not cleaned:
                continue
            key = self._dedupe_chunk_key(cleaned, message)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)

        return "\n\n".join(deduped).strip()

    def _is_bot_mention_only_content(self, content: str) -> bool:
        """
        True when content is only a ping to this bot (no actual text).

        These ping-only lines add noise in context blocks and are usually
        immediately followed by the real user question.
        """
        normalized = " ".join((content or "").split()).strip()
        if not normalized or not self._client or not getattr(self._client, "user", None):
            return False

        bot_user = self._client.user
        candidates: Set[str] = set()

        bot_id = str(getattr(bot_user, "id", "")).strip()
        if bot_id:
            candidates.add(f"<@{bot_id}>")
            candidates.add(f"<@!{bot_id}>")

        for raw_name in (
            getattr(bot_user, "name", None),
            getattr(bot_user, "global_name", None),
            getattr(bot_user, "display_name", None),
        ):
            name = str(raw_name or "").strip()
            if name:
                candidates.add(f"@{name}")

        return normalized in candidates

    def _message_to_archive_row(self, message: DiscordMessage) -> Dict[str, Any]:
        channel = message.channel
        guild = getattr(channel, "guild", None)
        author = message.author
        created_at = (
            float(message.created_at.timestamp())
            if getattr(message, "created_at", None) is not None
            else float(datetime.now().timestamp())
        )
        edited_at = (
            float(message.edited_at.timestamp())
            if getattr(message, "edited_at", None) is not None
            else None
        )
        normalized_content = self._materialize_message_text(message)

        return {
            "message_id": str(message.id),
            "guild_id": str(guild.id) if guild else None,
            "guild_name": guild.name if guild else None,
            "channel_id": str(channel.id),
            "channel_name": getattr(channel, "name", str(channel.id)),
            "thread_id": str(channel.id) if isinstance(channel, discord.Thread) else None,
            "author_id": str(author.id),
            "author_name": getattr(author, "name", ""),
            # Keep context/user attribution stable across servers by preferring
            # the global username (not guild-scoped nicknames).
            "author_display": getattr(author, "name", "") or str(author.id),
            "author_is_bot": bool(getattr(author, "bot", False)),
            "content": normalized_content,
            "attachments_json": [self._attachment_payload(att) for att in (message.attachments or [])],
            "created_at": created_at,
            "edited_at": edited_at,
            "deleted": False,
        }

    async def _start_archive_workers(self) -> None:
        """Start async archive workers after Discord has connected."""
        if not self._archive_db or not self._full_scrape_enabled():
            return
        self._archive_worker_stop.clear()

        if self._frontfill_task is None or self._frontfill_task.done():
            self._frontfill_task = asyncio.create_task(
                self._frontfill_loop(),
                name="discord-frontfill",
            )
        if self._backfill_enabled() and (self._backfill_task is None or self._backfill_task.done()):
            self._backfill_task = asyncio.create_task(
                self._backfill_loop(),
                name="discord-backfill",
            )

    async def _stop_archive_workers(self) -> None:
        """Stop and await async archive workers."""
        self._archive_worker_stop.set()
        tasks = [task for task in (self._frontfill_task, self._backfill_task) if task]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug("Discord archive worker shutdown error: %s", e)

        self._frontfill_task = None
        self._backfill_task = None
        self._archive_worker_stop.clear()

    async def _frontfill_loop(self) -> None:
        """Background forward-sync loop to keep channel heads up to date."""
        logger.info(
            "Discord front-filler started (interval=%ss, channels/tick=%s, pages/channel=%s)",
            self._full_scrape_interval_sec(),
            self._full_scrape_max_channels_per_tick(),
            self._full_scrape_max_pages_per_channel(),
        )
        try:
            while not self._archive_worker_stop.is_set():
                try:
                    await self._run_frontfill_tick()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug("Discord front-filler tick failed: %s", e)

                try:
                    await asyncio.wait_for(
                        self._archive_worker_stop.wait(),
                        timeout=self._full_scrape_interval_sec(),
                    )
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Discord front-filler stopped")

    async def _backfill_loop(self) -> None:
        """Low-priority backward-fill loop for older channel history."""
        logger.info(
            "Discord backfiller started (interval=%ss, channels/tick=%s, pages/channel=%s)",
            self._backfill_interval_sec(),
            self._backfill_max_channels_per_tick(),
            self._backfill_max_pages_per_channel(),
        )
        try:
            while not self._archive_worker_stop.is_set():
                try:
                    await self._run_backfill_tick()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug("Discord backfiller tick failed: %s", e)

                try:
                    await asyncio.wait_for(
                        self._archive_worker_stop.wait(),
                        timeout=self._backfill_interval_sec(),
                    )
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Discord backfiller stopped")

    @staticmethod
    def _pick_round_robin_batch(
        channels: List[Any],
        start_index: int,
        max_items: int,
    ) -> Tuple[List[Any], int]:
        if not channels or max_items <= 0:
            return [], 0
        n = len(channels)
        start = start_index % n
        take = min(max_items, n)
        ordered = channels[start:] + channels[:start]
        return ordered[:take], (start + take) % n

    def _collect_scrape_targets(self, include_threads: bool = False) -> List[Any]:
        """Collect readable Discord channels for archive workers."""
        if not DISCORD_AVAILABLE or not self._client:
            return []

        rows: List[Any] = []
        for guild in getattr(self._client, "guilds", []) or []:
            for channel in getattr(guild, "text_channels", []) or []:
                if self._is_channel_allowed(channel):
                    rows.append(channel)
            if include_threads:
                for thread in getattr(guild, "threads", []) or []:
                    if self._is_channel_allowed(thread):
                        rows.append(thread)

        # Stable ordering + de-dup by channel ID for fair round-robin selection.
        rows.sort(
            key=lambda ch: (
                str(getattr(getattr(ch, "guild", None), "id", "")),
                str(getattr(ch, "id", "")),
            )
        )
        deduped: List[Any] = []
        seen_ids: Set[str] = set()
        for channel in rows:
            channel_id = str(getattr(channel, "id", "")).strip()
            if not channel_id or channel_id in seen_ids:
                continue
            if not callable(getattr(channel, "history", None)):
                continue
            seen_ids.add(channel_id)
            deduped.append(channel)
        return deduped

    async def _run_frontfill_tick(self) -> None:
        if not self._archive_db:
            return
        targets = self._collect_scrape_targets(include_threads=self._full_scrape_include_threads())
        if not targets:
            return
        batch, self._frontfill_rr_index = self._pick_round_robin_batch(
            targets,
            self._frontfill_rr_index,
            self._full_scrape_max_channels_per_tick(),
        )
        for channel in batch:
            try:
                await self._sync_channel_forward(
                    channel,
                    max_pages=self._full_scrape_max_pages_per_channel(),
                    seed_limit=self._full_scrape_seed_limit(),
                    drain_all_pages=False,
                )
            except Exception as e:
                logger.debug(
                    "Discord front-filler channel sync failed (%s): %s",
                    getattr(channel, "id", "unknown"),
                    e,
                )

    async def _run_backfill_tick(self) -> None:
        if not self._archive_db or not self._backfill_enabled():
            return
        targets = self._collect_scrape_targets(include_threads=self._full_scrape_include_threads())
        if not targets:
            return
        batch, self._backfill_rr_index = self._pick_round_robin_batch(
            targets,
            self._backfill_rr_index,
            self._backfill_max_channels_per_tick(),
        )
        for channel in batch:
            try:
                await self._sync_channel_backfill(
                    channel,
                    max_pages=self._backfill_max_pages_per_channel(),
                )
            except Exception as e:
                logger.debug(
                    "Discord backfiller channel sync failed (%s): %s",
                    getattr(channel, "id", "unknown"),
                    e,
                )

    async def _sync_channel_forward(
        self,
        channel: Any,
        *,
        max_pages: int,
        seed_limit: Optional[int] = None,
        drain_all_pages: bool = False,
    ) -> int:
        """
        Sync newer messages for one channel using the latest cursor.

        Returns number of messages upserted.
        """
        if not self._archive_db or not DISCORD_AVAILABLE:
            return 0
        if not self._is_channel_allowed(channel):
            return 0

        channel_id = str(getattr(channel, "id", "")).strip()
        if not channel_id or channel_id in self._active_history_channels:
            return 0
        self._active_history_channels.add(channel_id)

        try:
            cursor = str(self._archive_db.get_channel_cursor(channel_id) or "").strip()
            if not cursor:
                limit = max(1, int(seed_limit or self._full_scrape_seed_limit()))
                seeded = [m async for m in channel.history(limit=limit, oldest_first=True)]
                for hist_msg in seeded:
                    self._archive_db.upsert_message(self._message_to_archive_row(hist_msg))
                if seeded:
                    logger.info(
                        "Discord front-filler seed sync complete (channel=%s, messages=%s, seed_limit=%s)",
                        channel_id,
                        len(seeded),
                        limit,
                    )
                return len(seeded)

            after_obj = discord.Object(id=int(cursor)) if cursor.isdigit() else None
            if after_obj is None:
                # Invalid cursor fallback: refresh a bounded recent window.
                limit = max(1, int(seed_limit or self._full_scrape_seed_limit()))
                refreshed = [m async for m in channel.history(limit=limit, oldest_first=True)]
                for hist_msg in refreshed:
                    self._archive_db.upsert_message(self._message_to_archive_row(hist_msg))
                if refreshed:
                    logger.info(
                        "Discord front-filler cursor refresh complete (channel=%s, messages=%s, refresh_limit=%s)",
                        channel_id,
                        len(refreshed),
                        limit,
                    )
                return len(refreshed)

            pages_left = None if drain_all_pages else max(1, int(max_pages))
            total = 0
            page_count = 0
            progress_every = self._scrape_progress_every_pages()
            while True:
                kwargs = {"limit": 100, "oldest_first": True, "after": after_obj}
                batch = [m async for m in channel.history(**kwargs)]
                if not batch:
                    break
                for hist_msg in batch:
                    self._archive_db.upsert_message(self._message_to_archive_row(hist_msg))
                total += len(batch)
                page_count += 1

                latest_id = str(getattr(batch[-1], "id", "")).strip()
                if page_count % progress_every == 0:
                    logger.info(
                        "Discord front-filler progress (channel=%s, pages=%s, messages=%s, latest_message_id=%s)",
                        channel_id,
                        page_count,
                        total,
                        latest_id or "n/a",
                    )
                if not latest_id or not latest_id.isdigit():
                    break
                after_obj = discord.Object(id=int(latest_id))

                if len(batch) < 100:
                    break
                if pages_left is not None:
                    pages_left -= 1
                    if pages_left <= 0:
                        break
            if page_count > 0:
                logger.info(
                    "Discord front-filler channel sync complete (channel=%s, pages=%s, messages=%s)",
                    channel_id,
                    page_count,
                    total,
                )
            return total
        finally:
            self._active_history_channels.discard(channel_id)

    async def _sync_channel_backfill(self, channel: Any, *, max_pages: int) -> int:
        """
        Sync older messages for one channel using a backward cursor.

        Returns number of messages upserted.
        """
        if not self._archive_db or not DISCORD_AVAILABLE:
            return 0
        if not self._is_channel_allowed(channel):
            return 0

        channel_id = str(getattr(channel, "id", "")).strip()
        if not channel_id or channel_id in self._active_history_channels:
            return 0
        self._active_history_channels.add(channel_id)

        try:
            state = self._archive_db.get_backfill_state(channel_id)
            if bool(state.get("complete")):
                return 0

            oldest_message_id = str(state.get("oldest_message_id") or "").strip()
            oldest_created_at = state.get("oldest_created_at")
            if not oldest_message_id:
                oldest_row = self._archive_db.get_oldest_message(channel_id)
                if not oldest_row:
                    return 0
                oldest_message_id = str(oldest_row.get("message_id") or "").strip()
                oldest_created_at = oldest_row.get("created_at")
                if not oldest_message_id:
                    return 0
                self._archive_db.upsert_backfill_state(
                    channel_id,
                    oldest_message_id=oldest_message_id,
                    oldest_created_at=oldest_created_at,
                    complete=False,
                )

            if not oldest_message_id.isdigit():
                self._archive_db.mark_backfill_complete(channel_id, complete=True)
                return 0

            before_obj = discord.Object(id=int(oldest_message_id))
            pages_left = max(1, int(max_pages))
            total = 0
            page_count = 0
            reached_start = False
            progress_every = self._scrape_progress_every_pages()

            while pages_left > 0:
                batch = [m async for m in channel.history(limit=100, oldest_first=False, before=before_obj)]
                if not batch:
                    reached_start = True
                    break

                for hist_msg in batch:
                    self._archive_db.upsert_message(self._message_to_archive_row(hist_msg))
                total += len(batch)
                page_count += 1

                oldest_msg = batch[-1]
                next_oldest_id = str(getattr(oldest_msg, "id", "")).strip()
                next_oldest_created = (
                    float(oldest_msg.created_at.timestamp())
                    if getattr(oldest_msg, "created_at", None) is not None
                    else None
                )
                if next_oldest_id:
                    self._archive_db.upsert_backfill_state(
                        channel_id,
                        oldest_message_id=next_oldest_id,
                        oldest_created_at=next_oldest_created,
                        complete=False,
                    )
                if page_count % progress_every == 0:
                    logger.info(
                        "Discord backfiller progress (channel=%s, pages=%s, messages=%s, oldest_message_id=%s)",
                        channel_id,
                        page_count,
                        total,
                        next_oldest_id or "n/a",
                    )
                if not next_oldest_id or not next_oldest_id.isdigit():
                    reached_start = True
                    break
                if next_oldest_id == oldest_message_id:
                    reached_start = True
                    break
                oldest_message_id = next_oldest_id
                before_obj = discord.Object(id=int(oldest_message_id))

                if len(batch) < 100:
                    reached_start = True
                    break
                pages_left -= 1

            if reached_start:
                self._archive_db.mark_backfill_complete(channel_id, complete=True)
            if page_count > 0:
                logger.info(
                    "Discord backfiller channel sync complete (channel=%s, pages=%s, messages=%s, complete=%s)",
                    channel_id,
                    page_count,
                    total,
                    reached_start,
                )
            return total
        finally:
            self._active_history_channels.discard(channel_id)

    async def _bootstrap_channel_archive(self, channel: Any) -> None:
        """
        One-time channel catch-up for this process.

        - If we have no cursor, seed with a recent window.
        - If we have a cursor, fetch forward in 100-message pages.
        """
        if not self._archive_db or not DISCORD_AVAILABLE:
            return

        channel_id = str(getattr(channel, "id", ""))
        if not channel_id or channel_id in self._bootstrapped_channels:
            return

        self._bootstrapped_channels.add(channel_id)
        try:
            await self._sync_channel_forward(
                channel,
                max_pages=self._full_scrape_max_pages_per_channel(),
                seed_limit=max(self._fresh_context_limit(), 20),
                drain_all_pages=True,
            )
        except Exception as e:
            logger.debug("Discord channel bootstrap failed (%s): %s", channel_id, e)

    async def _archive_message(self, message: DiscordMessage) -> None:
        """Persist one Discord message into local SQLite archive."""
        if not self._archive_db:
            return
        if not self._is_channel_allowed(message.channel):
            return

        # Bootstrap failures should never drop the live message that triggered
        # this handler.
        try:
            await self._bootstrap_channel_archive(message.channel)
        except Exception as e:
            logger.debug("Discord channel bootstrap failed while archiving: %s", e)

        try:
            self._archive_db.upsert_message(self._message_to_archive_row(message))
        except Exception as e:
            logger.debug("Discord archive upsert failed: %s", e)

    async def _archive_message_edit(self, before: DiscordMessage, after: DiscordMessage) -> None:
        """Persist one Discord message edit as both change-log and latest row."""
        if not self._archive_db:
            return
        if not self._is_channel_allowed(after.channel):
            return

        try:
            await self._bootstrap_channel_archive(after.channel)
        except Exception as e:
            logger.debug("Discord channel bootstrap failed while archiving edit: %s", e)

        try:
            before_row = self._message_to_archive_row(before)
            after_row = self._message_to_archive_row(after)
            self._archive_db.record_message_edit(
                message_id=after_row["message_id"],
                channel_id=after_row["channel_id"],
                guild_id=after_row.get("guild_id"),
                author_id=after_row.get("author_id"),
                author_name=after_row.get("author_name"),
                author_display=after_row.get("author_display"),
                author_is_bot=bool(after_row.get("author_is_bot")),
                original_created_at=after_row.get("created_at"),
                changed_at=after_row.get("edited_at"),
                before_content=before_row.get("content"),
                after_content=after_row.get("content"),
            )
            self._archive_db.upsert_message(after_row)
        except Exception as e:
            logger.debug("Discord archive edit upsert failed: %s", e)

    def reset_channel_context(self, channel_id: str) -> None:
        """
        Clear channel-level context state so the next turn is a fresh window.

        Used by `/new` and `/reset`.
        """
        ch_id = str(channel_id or "").strip()
        if not ch_id:
            return
        self._context_header_sent_channels.discard(ch_id)
        self._force_fresh_context_channels.add(ch_id)
        if not self._archive_db:
            return
        try:
            self._archive_db.clear_turn_anchor(ch_id)
        except Exception as e:
            logger.debug("Discord turn-anchor reset failed (%s): %s", ch_id, e)

    @staticmethod
    def _format_archive_history_line(msg: Dict[str, Any]) -> Tuple[str, str]:
        """
        Format one archived message as:
            header: DD/MM/YYYY HH
            line:   MM:SS <username>: content
        """
        ts = float(msg.get("created_at") or 0)
        dt = datetime.fromtimestamp(ts) if ts else datetime.now()
        hour_header = dt.strftime("%d/%m/%Y %H")
        minute_second = dt.strftime("%M:%S")
        author = msg.get("author_display") or msg.get("author_name") or msg.get("author_id") or "unknown"
        content = " ".join((msg.get("content") or "").split())
        if not content:
            content = "[non-text message]"
        # if len(content) > 220:
        #     content = content[:217] + "..."
        return hour_header, f"{minute_second} <{author}>: {content}"

    @staticmethod
    def _format_archive_change_line(change: Dict[str, Any]) -> Tuple[str, str]:
        """
        Format one archived message change as:
            header: DD/MM/YYYY HH
            line:   MM:SS <username>: before -> after
        """
        ts = float(
            change.get("original_created_at")
            or change.get("changed_at")
            or 0
        )
        dt = datetime.fromtimestamp(ts) if ts else datetime.now()
        hour_header = dt.strftime("%d/%m/%Y %H")
        minute_second = dt.strftime("%M:%S")
        author = (
            change.get("author_display")
            or change.get("author_name")
            or change.get("author_id")
            or "unknown"
        )
        before = " ".join((change.get("before_content") or "").split())
        after = " ".join((change.get("after_content") or "").split())
        change_type = str(change.get("change_type") or "").strip().lower()

        if not before:
            before = "[non-text message]"
        if change_type == "delete":
            after = "[Deleted]"
        elif not after:
            after = "[non-text message]"

        if len(before) > 120:
            before = before[:117] + "..."
        if len(after) > 120:
            after = after[:117] + "..."

        return hour_header, f"{minute_second} <{author}>: {before} -> {after}"

    def _render_context_block(
        self,
        rows: List[Dict[str, Any]],
        channel_label: str,
        _mode_label: str,
        include_channel_label: bool = True,
        changes: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not rows and not changes:
            return ""

        lines: List[str] = []
        last_hour: Optional[str] = None
        for row in rows:
            hour_header, line = self._format_archive_history_line(row)
            if hour_header != last_hour:
                lines.append(hour_header)
                last_hour = hour_header
            lines.append(line)

        if changes:
            if lines:
                lines.append("")
            lines.append("[Changes]")
            change_hour: Optional[str] = None
            for change in changes:
                hour_header, line = self._format_archive_change_line(change)
                if hour_header != change_hour:
                    lines.append(hour_header)
                    change_hour = hour_header
                lines.append(line)

        header = (
            f"[Discord context | {channel_label}]"
            if include_channel_label
            else "[Discord context]"
        )
        block = header + "\n" + "\n".join(lines)
        max_chars = self._context_max_chars()
        if len(block) <= max_chars:
            return block

        # Keep the most recent portion plus the header.
        budget = max(500, max_chars - len(header) - 32)
        body = "\n".join(lines)
        if len(body) > budget:
            body = "...[context truncated]...\n" + body[-budget:]
        return header + "\n" + body

    async def _build_recent_channel_context(self, message: DiscordMessage) -> Tuple[str, bool, str]:
        """
        Build channel-local context with delta-aware auto-reset hints.

        Returns:
            (context_block, force_auto_reset, auto_reset_reason)
        """
        fresh_limit = self._fresh_context_limit()
        window_limit = max(1, fresh_limit)
        channel_id = str(message.channel.id)

        if not self._archive_db:
            # Fallback path (archive disabled): hit Discord history API directly.
            try:
                rows = []
                async for msg in message.channel.history(limit=min(window_limit, 50), oldest_first=True):
                    rows.append(
                        {
                            "message_id": str(msg.id),
                            "created_at": float(msg.created_at.timestamp()),
                            "author_name": getattr(msg.author, "name", None) or "unknown",
                            "content": msg.content or "",
                        }
                    )
            except Exception as e:
                logger.debug("Discord channel context fetch failed: %s", e)
                return "", False, ""
            ch_name = getattr(message.channel, "name", str(message.channel.id))
            guild = getattr(message.channel, "guild", None)
            label = f"{guild.name} / #{ch_name}" if guild else ch_name
            include_channel_label = channel_id not in self._context_header_sent_channels
            block = self._render_context_block(
                rows,
                label,
                f"fallback_last_{window_limit}",
                include_channel_label=include_channel_label,
            )
            if block and include_channel_label:
                self._context_header_sent_channels.add(channel_id)
            return block, False, ""

        force_fresh_window = channel_id in self._force_fresh_context_channels
        anchor = None if force_fresh_window else self._archive_db.get_turn_anchor(channel_id)
        threshold = self._delta_reset_threshold()

        force_auto_reset = False
        reset_reason = ""
        delta = 0
        if anchor:
            delta = self._archive_db.count_new_non_bot_messages(channel_id, anchor)
            if delta > threshold:
                force_auto_reset = True
                reset_reason = f"channel delta {delta} exceeded threshold {threshold}"

        # Context strategy:
        # - Fresh sessions/reset sessions: inject last-N (fresh window).
        # - Normal follow-up turns: inject only messages after last anchor.
        if anchor:
            if force_auto_reset:
                rows = self._archive_db.list_recent_messages(
                    channel_id=channel_id,
                    limit=window_limit,
                    include_bots=False,
                )
            else:
                # Delta-only follow-up mode:
                # same threshold drives both reset behavior and max delta payload.
                rows = self._archive_db.list_messages_after(
                    channel_id=channel_id,
                    after_message_id=anchor,
                    limit=threshold,
                    include_bots=False,
                )
        else:
            rows = self._archive_db.list_recent_messages(
                channel_id=channel_id,
                limit=window_limit,
                include_bots=False,
            )

        if anchor:
            mode_label = f"delta_after_anchor_{delta}"
            if force_auto_reset:
                mode_label = f"fresh_last_{window_limit}_reset_delta_{delta}"
        else:
            mode_label = f"fresh_last_{window_limit}"
            if force_fresh_window:
                mode_label = f"fresh_last_{window_limit}_forced_reset"

        # Optional legacy fallback:
        # if delta mode yields nothing except the current turn (removed above),
        # some deployments prefer fresh last-N for extra context.
        if (
            not rows
            and anchor
            and not force_auto_reset
            and self._empty_delta_fallback_enabled()
        ):
            rows = self._archive_db.list_recent_messages(
                channel_id=channel_id,
                limit=window_limit,
                include_bots=False,
            )
            mode_label = f"fresh_last_{window_limit}_empty_delta"

        # Drop ping-only mentions to the bot from context; they are typically
        # noise and can appear as "duplicate-looking" turns.
        rows = [
            row
            for row in rows
            if not self._is_bot_mention_only_content(str(row.get("content") or ""))
        ]

        change_rows: List[Dict[str, Any]] = []
        if anchor and hasattr(self._archive_db, "list_changes_since_anchor"):
            try:
                change_rows = self._archive_db.list_changes_since_anchor(
                    channel_id=channel_id,
                    anchor_message_id=anchor,
                    limit=threshold,
                    include_bots=False,
                )
            except Exception as e:
                logger.debug("Discord change-log query failed: %s", e)

        # Advance anchor for the next responding turn.
        self._archive_db.set_turn_anchor(channel_id, str(message.id))
        if force_fresh_window:
            self._force_fresh_context_channels.discard(channel_id)

        if not rows and not change_rows:
            return "", force_auto_reset, reset_reason

        ch_name = getattr(message.channel, "name", str(message.channel.id))
        guild = getattr(message.channel, "guild", None)
        label = f"{guild.name} / #{ch_name}" if guild else ch_name
        include_channel_label = anchor is None
        block = self._render_context_block(
            rows,
            label,
            mode_label,
            include_channel_label=include_channel_label,
            changes=change_rows,
        )
        if block and include_channel_label:
            self._context_header_sent_channels.add(channel_id)
        return block, force_auto_reset, reset_reason
    
    async def _resolve_allowed_usernames(self) -> None:
        """
        Resolve non-numeric entries in DISCORD_ALLOWED_USERS to Discord user IDs.

        Users can specify usernames (e.g. "teknium") or display names instead of
        raw numeric IDs.  After resolution, the env var and internal set are updated
        so authorization checks work with IDs only.
        """
        if not self._allowed_user_ids or not self._client:
            return

        numeric_ids = set()
        to_resolve = set()

        for entry in self._allowed_user_ids:
            if entry.isdigit():
                numeric_ids.add(entry)
            else:
                to_resolve.add(entry.lower())

        if not to_resolve:
            return

        print(f"[{self.name}] Resolving {len(to_resolve)} username(s): {', '.join(to_resolve)}")
        resolved_count = 0

        for guild in self._client.guilds:
            # Fetch full member list (requires members intent)
            try:
                members = guild.members
                if len(members) < guild.member_count:
                    members = [m async for m in guild.fetch_members(limit=None)]
            except Exception as e:
                logger.warning("Failed to fetch members for guild %s: %s", guild.name, e)
                continue

            for member in members:
                name_lower = member.name.lower()
                display_lower = member.display_name.lower()
                global_lower = (member.global_name or "").lower()

                matched = name_lower in to_resolve or display_lower in to_resolve or global_lower in to_resolve
                if matched:
                    uid = str(member.id)
                    numeric_ids.add(uid)
                    resolved_count += 1
                    matched_name = name_lower if name_lower in to_resolve else (
                        display_lower if display_lower in to_resolve else global_lower
                    )
                    to_resolve.discard(matched_name)
                    print(f"[{self.name}] Resolved '{matched_name}' -> {uid} ({member.name}#{member.discriminator})")

            if not to_resolve:
                break

        if to_resolve:
            print(f"[{self.name}] Could not resolve usernames: {', '.join(to_resolve)}")

        # Update internal set and env var so gateway auth checks use IDs
        self._allowed_user_ids = numeric_ids
        os.environ["DISCORD_ALLOWED_USERS"] = ",".join(sorted(numeric_ids))
        if resolved_count:
            print(f"[{self.name}] Updated DISCORD_ALLOWED_USERS with {resolved_count} resolved ID(s)")

    def format_message(self, content: str) -> str:
        """
        Format message for Discord.
        
        Discord uses its own markdown variant.
        """
        # Discord markdown is fairly standard, no special escaping needed
        return content

    async def _guard_slash_channel(self, interaction: discord.Interaction) -> bool:
        """Reject slash commands outside configured guild/channel allowlists."""
        channel = getattr(interaction, "channel", None)
        if self._is_channel_allowed(channel):
            return True
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "This channel is not in the allowed Discord scope.",
                    ephemeral=True,
                )
            else:
                await interaction.followup.send(
                    "This channel is not in the allowed Discord scope.",
                    ephemeral=True,
                )
        except Exception:
            pass
        return False
    
    def _register_slash_commands(self) -> None:
        """Register Discord slash commands on the command tree."""
        if not self._client:
            return

        tree = self._client.tree

        @tree.command(name="ask", description="Ask Hermes a question")
        @discord.app_commands.describe(question="Your question for Hermes")
        async def slash_ask(interaction: discord.Interaction, question: str):
            if not await self._guard_slash_channel(interaction):
                return
            await interaction.response.defer()
            event = self._build_slash_event(interaction, question)
            await self.handle_message(event)
            # The response is sent via the normal send() flow
            # Send a followup to close the interaction if needed
            try:
                await interaction.followup.send("Processing complete~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="new", description="Start a new conversation")
        async def slash_new(interaction: discord.Interaction):
            if not await self._guard_slash_channel(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/reset")
            await self.handle_message(event)
            try:
                await interaction.followup.send("New conversation started~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="reset", description="Reset your Hermes session")
        async def slash_reset(interaction: discord.Interaction):
            if not await self._guard_slash_channel(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/reset")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Session reset~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="model", description="Show or change the model")
        @discord.app_commands.describe(name="Model name (e.g. anthropic/claude-sonnet-4). Leave empty to see current.")
        async def slash_model(interaction: discord.Interaction, name: str = ""):
            if not await self._guard_slash_channel(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, f"/model {name}".strip())
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="personality", description="Set a personality")
        @discord.app_commands.describe(name="Personality name. Leave empty to list available.")
        async def slash_personality(interaction: discord.Interaction, name: str = ""):
            if not await self._guard_slash_channel(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, f"/personality {name}".strip())
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="retry", description="Retry your last message")
        async def slash_retry(interaction: discord.Interaction):
            if not await self._guard_slash_channel(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/retry")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Retrying~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="undo", description="Remove the last exchange")
        async def slash_undo(interaction: discord.Interaction):
            if not await self._guard_slash_channel(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/undo")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="status", description="Show Hermes session status")
        async def slash_status(interaction: discord.Interaction):
            if not await self._guard_slash_channel(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/status")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Status sent~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="sethome", description="Set this chat as the home channel")
        async def slash_sethome(interaction: discord.Interaction):
            if not await self._guard_slash_channel(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/sethome")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Done~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

        @tree.command(name="stop", description="Stop the running Hermes agent")
        async def slash_stop(interaction: discord.Interaction):
            if not await self._guard_slash_channel(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            event = self._build_slash_event(interaction, "/stop")
            await self.handle_message(event)
            try:
                await interaction.followup.send("Stop requested~", ephemeral=True)
            except Exception as e:
                logger.debug("Discord followup failed: %s", e)

    def _build_slash_event(self, interaction: discord.Interaction, text: str) -> MessageEvent:
        """Build a MessageEvent from a Discord slash command interaction."""
        is_dm = isinstance(interaction.channel, discord.DMChannel)
        chat_type = "dm" if is_dm else "group"
        chat_name = ""
        if not is_dm and hasattr(interaction.channel, "name"):
            chat_name = interaction.channel.name
            if hasattr(interaction.channel, "guild") and interaction.channel.guild:
                chat_name = f"{interaction.channel.guild.name} / #{chat_name}"

        source = self.build_source(
            chat_id=str(interaction.channel_id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(interaction.user.id),
            user_name=interaction.user.name,
        )

        msg_type = MessageType.COMMAND if text.startswith("/") else MessageType.TEXT
        return MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=interaction,
        )

    async def send_exec_approval(
        self, chat_id: str, command: str, approval_id: str
    ) -> SendResult:
        """
        Send a button-based exec approval prompt for a dangerous command.

        Returns SendResult. The approval is resolved when a user clicks a button.
        """
        if not self._client or not DISCORD_AVAILABLE:
            return SendResult(success=False, error="Not connected")

        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            if isinstance(channel, discord.DMChannel) and not self._dms_enabled():
                return SendResult(
                    success=False,
                    error="Discord DMs are disabled (DISCORD_ENABLE_DMS=false)",
                )

            embed = discord.Embed(
                title="Command Approval Required",
                description=f"```\n{command[:500]}\n```",
                color=discord.Color.orange(),
            )
            embed.set_footer(text=f"Approval ID: {approval_id}")

            view = ExecApprovalView(
                approval_id=approval_id,
                allowed_user_ids=self._allowed_user_ids,
            )

            msg = await channel.send(embed=embed, view=view)
            return SendResult(success=True, message_id=str(msg.id))

        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def _handle_message(self, message: DiscordMessage) -> None:
        """Handle incoming Discord messages."""
        if not self._is_channel_allowed(message.channel):
            logger.debug("Discord message ignored (channel/guild not allowed): %s", message.channel.id)
            return

        # Persist all visible channel traffic before any mention/free-response gating.
        await self._archive_message(message)
        incoming_text = message.content or ""

        # In server channels (not DMs), require the bot to be @mentioned
        # UNLESS the channel is in the free-response list.
        #
        # Config:
        #   DISCORD_FREE_RESPONSE_CHANNELS: Comma-separated channel IDs where the
        #       bot responds to every message without needing a mention.
        #   DISCORD_REQUIRE_MENTION: Set to "false" to disable mention requirement
        #       globally (all channels become free-response). Default: "true".
        
        if not isinstance(message.channel, discord.DMChannel):
            # Check if this channel is in the free-response list
            free_channels_raw = os.getenv("DISCORD_FREE_RESPONSE_CHANNELS", "")
            free_channels = {ch.strip() for ch in free_channels_raw.split(",") if ch.strip()}
            channel_id = str(message.channel.id)
            
            # Global override: if DISCORD_REQUIRE_MENTION=false, all channels are free
            require_mention = os.getenv("DISCORD_REQUIRE_MENTION", "true").lower() not in ("false", "0", "no")
            
            is_free_channel = channel_id in free_channels
            
            if require_mention and not is_free_channel:
                # Must be @mentioned to respond
                if self._client.user not in message.mentions:
                    return  # Silently ignore messages that don't mention the bot
            
            # Strip the bot mention from the message text so the agent sees clean input
            if self._client.user and self._client.user in message.mentions:
                incoming_text = incoming_text.replace(f"<@{self._client.user.id}>", "").strip()
                incoming_text = incoming_text.replace(f"<@!{self._client.user.id}>", "").strip()

        incoming_text = self._replace_user_mentions(incoming_text, message)
        command_probe_text = incoming_text
        incoming_text = self._materialize_message_text(
            message,
            base_content=incoming_text,
        )
        
        # Determine message type
        msg_type = MessageType.TEXT
        if command_probe_text.startswith("/"):
            msg_type = MessageType.COMMAND
        elif message.attachments:
            # Check attachment types
            for att in message.attachments:
                if att.content_type:
                    if att.content_type.startswith("image/"):
                        msg_type = MessageType.PHOTO
                    elif att.content_type.startswith("video/"):
                        msg_type = MessageType.VIDEO
                    elif att.content_type.startswith("audio/"):
                        msg_type = MessageType.AUDIO
                    else:
                        msg_type = MessageType.DOCUMENT
                    break
        
        # Determine chat type
        if isinstance(message.channel, discord.DMChannel):
            chat_type = "dm"
            chat_name = message.author.name
        elif isinstance(message.channel, discord.Thread):
            chat_type = "thread"
            chat_name = message.channel.name
        else:
            chat_type = "group"  # Treat server channels as groups
            chat_name = getattr(message.channel, "name", str(message.channel.id))
            if hasattr(message.channel, "guild") and message.channel.guild:
                chat_name = f"{message.channel.guild.name} / #{chat_name}"
        
        # Get thread ID if in a thread
        thread_id = None
        if isinstance(message.channel, discord.Thread):
            thread_id = str(message.channel.id)
        
        # Build source
        source = self.build_source(
            chat_id=str(message.channel.id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(message.author.id),
            user_name=message.author.name,
            thread_id=thread_id,
        )
        
        # Build media URLs -- download image attachments to local cache so the
        # vision tool can access them reliably (Discord CDN URLs can expire).
        media_urls = []
        media_types = []
        for att in message.attachments:
            content_type = att.content_type or "unknown"
            if content_type.startswith("image/"):
                if self._is_gif_attachment(content_type, getattr(att, "filename", "")):
                    continue
                try:
                    # Determine extension from content type (image/png -> .png)
                    ext = "." + content_type.split("/")[-1].split(";")[0]
                    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                        ext = ".jpg"
                    cached_path = await cache_image_from_url(att.url, ext=ext)
                    media_urls.append(cached_path)
                    media_types.append(f"{content_type};source_url={att.url}")
                    print(f"[Discord] Cached user image: {cached_path}", flush=True)
                except Exception as e:
                    print(f"[Discord] Failed to cache image attachment: {e}", flush=True)
                    # Fall back to the CDN URL if caching fails
                    media_urls.append(att.url)
                    media_types.append(f"{content_type};source_url={att.url}")
            elif content_type.startswith("audio/"):
                try:
                    ext = "." + content_type.split("/")[-1].split(";")[0]
                    if ext not in (".ogg", ".mp3", ".wav", ".webm", ".m4a"):
                        ext = ".ogg"
                    cached_path = await cache_audio_from_url(att.url, ext=ext)
                    media_urls.append(cached_path)
                    media_types.append(content_type)
                    print(f"[Discord] Cached user audio: {cached_path}", flush=True)
                except Exception as e:
                    print(f"[Discord] Failed to cache audio attachment: {e}", flush=True)
                    media_urls.append(att.url)
                    media_types.append(content_type)
            else:
                # Other attachments: keep the original URL
                media_urls.append(att.url)
                media_types.append(content_type)
        for emoji_name, emoji_url in self._extract_static_custom_emojis(incoming_text):
            try:
                cached_path = await cache_image_from_url(emoji_url, ext=".png")
                media_urls.append(cached_path)
                media_types.append(
                    f"image/x-discord-emoji;name={emoji_name};source_url={emoji_url}"
                )
            except Exception as e:
                logger.debug("Discord custom emoji cache failed (%s): %s", emoji_name, e)
                media_urls.append(emoji_url)
                media_types.append(
                    f"image/x-discord-emoji;name={emoji_name};source_url={emoji_url}"
                )

        has_current_image = any(
            (mt or "").lower().startswith("image/") or (mt or "").lower().startswith("image/x-discord-emoji")
            for mt in media_types
        )
        carryover_image_items: List[Dict[str, str]] = []
        if not has_current_image and incoming_text.strip():
            carryover_image_items = self._collect_recent_channel_image_items(
                message,
                max_images=self._followup_image_max_count(),
            )
            if carryover_image_items:
                logger.debug(
                    "Discord follow-up turn: carrying %d prior channel image(s) for user %s in channel %s",
                    len(carryover_image_items),
                    message.author.id,
                    message.channel.id,
                )

        extra_context, force_auto_reset, auto_reset_reason = await self._build_recent_channel_context(message)
        
        event = MessageEvent(
            text=incoming_text,
            message_type=msg_type,
            source=source,
            raw_message=message,
            message_id=str(message.id),
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=str(message.reference.message_id) if message.reference else None,
            timestamp=message.created_at,
            extra_context=extra_context,
            carryover_image_items=carryover_image_items,
            force_auto_reset=force_auto_reset,
            auto_reset_reason=auto_reset_reason,
        )
        
        await self.handle_message(event)


# ---------------------------------------------------------------------------
# Discord UI Components (outside the adapter class)
# ---------------------------------------------------------------------------

if DISCORD_AVAILABLE:

    class ExecApprovalView(discord.ui.View):
        """
        Interactive button view for exec approval of dangerous commands.

        Shows three buttons: Allow Once (green), Always Allow (blue), Deny (red).
        Only users in the allowed list can click. The view times out after 5 minutes.
        """

        def __init__(self, approval_id: str, allowed_user_ids: set):
            super().__init__(timeout=300)  # 5-minute timeout
            self.approval_id = approval_id
            self.allowed_user_ids = allowed_user_ids
            self.resolved = False

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            """Verify the user clicking is authorized."""
            if not self.allowed_user_ids:
                return True  # No allowlist = anyone can approve
            return str(interaction.user.id) in self.allowed_user_ids

        async def _resolve(
            self, interaction: discord.Interaction, action: str, color: discord.Color
        ):
            """Resolve the approval and update the message."""
            if self.resolved:
                await interaction.response.send_message(
                    "This approval has already been resolved~", ephemeral=True
                )
                return

            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    "You're not authorized to approve commands~", ephemeral=True
                )
                return

            self.resolved = True

            # Update the embed with the decision
            embed = interaction.message.embeds[0] if interaction.message.embeds else None
            if embed:
                embed.color = color
                embed.set_footer(text=f"{action} by {interaction.user.display_name}")

            # Disable all buttons
            for child in self.children:
                child.disabled = True

            await interaction.response.edit_message(embed=embed, view=self)

            # Store the approval decision
            try:
                from tools.approval import approve_permanent
                if action == "allow_once":
                    pass  # One-time approval handled by gateway
                elif action == "allow_always":
                    approve_permanent(self.approval_id)
            except ImportError:
                pass

        @discord.ui.button(label="Allow Once", style=discord.ButtonStyle.green)
        async def allow_once(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "allow_once", discord.Color.green())

        @discord.ui.button(label="Always Allow", style=discord.ButtonStyle.blurple)
        async def allow_always(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "allow_always", discord.Color.blue())

        @discord.ui.button(label="Deny", style=discord.ButtonStyle.red)
        async def deny(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ):
            await self._resolve(interaction, "deny", discord.Color.red())

        async def on_timeout(self):
            """Handle view timeout -- disable buttons and mark as expired."""
            self.resolved = True
            for child in self.children:
                child.disabled = True
