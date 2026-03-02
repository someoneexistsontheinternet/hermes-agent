# Discord Gateway Update Notes (2026-03-03)

This document summarizes the code currently staged in this branch.

## Major Changes

1. Discord channel archive subsystem (SQLite + FTS5)
   - Added a persistent local archive in `gateway/discord_archive.py`.
   - Captures message rows, edit/delete change history, per-channel cursors, and per-channel turn anchors.
   - Added forward sync ("frontfill") and backward sync ("backfill") primitives for channel history.
   - Added FTS5 indexing and trigger maintenance for searchable message content.

2. Discord adapter context pipeline overhaul
   - `gateway/platforms/discord.py` now builds channel-local context blocks from archived messages.
   - Added delta-based context mode: follow-up turns use messages after the previous turn anchor.
   - Added automatic channel-session reset hint when channel delta exceeds threshold.
   - Added optional fallback from empty delta to fresh last-N context.
   - Added support to include edit/delete changes in context via `[Changes]` blocks.
   - Added channel/guild allowlist enforcement and DM enable/disable gate in adapter paths.

3. Background archive workers for Discord
   - Added async frontfill and backfill loops in `gateway/platforms/discord.py`.
   - Added round-robin channel batching and per-tick limits.
   - Added configurable scrape controls (intervals, page limits, include threads, progress cadence).

4. Multimodal image handling improvements in gateway runner
   - `gateway/run.py` now supports direct multimodal user payloads for model families known to accept image parts.
   - Added payload URL normalization to base64 data URLs (local cached files preferred).
   - Added historical multimodal content sanitization to avoid invalid/expired image URLs.
   - Added follow-up image carryover support from recent Discord channel activity.

5. New Discord archive search tool
   - Added `tools/discord_search_tool.py`, registered via `model_tools.py` and `toolsets.py`.
   - Supports FTS query mode and read-only SQL mode.
   - SQL mode includes statement/class restrictions plus SQLite authorizer guards.
   - Supports filters by guild/channel/author and time windows, with optional context windows around hits.

6. Runtime model routing extraction + passthrough
   - Added `model_runtime_config.py` to resolve model/base URL/provider/extra_body from `config.yaml`.
   - Integrated into gateway runner and cron scheduler.
   - Added `model.extra_body` pass-through in CLI and agent API call composition.

7. Session and persistence reliability updates
   - `gateway/session.py`: added legacy transcript fallback logic that can prefer richer fallback transcript paths.
   - `run_agent.py` and `hermes_state.py`: improved serialization/deserialization for non-string (multimodal) content.
   - `gateway/run.py`: disables duplicate session-db writes from nested agent path (`session_db_writes=False`) and deduplicates adjacent historical tool/assistant rows.

## Minor Changes

1. Config and path behavior
   - `gateway/config.py` now consistently honors `HERMES_HOME` for defaults and config file locations.
   - Added bridge from `config.yaml` `gateway.discord` keys into gateway platform config extras.
   - `cli-config.yaml.example` now documents Discord gateway context/archive settings.

2. Discord behavior and formatting details
   - Better mention replacement/text materialization and static custom emoji extraction.
   - Improved channel readability checks (view + read history) in `gateway/channel_directory.py`.
   - Added slash-command channel guard for out-of-scope channels.
   - `/new`/`/reset` now clear Discord channel turn anchor via adapter hook.

3. Auth and environment handling
   - Gateway `.env` loading now uses override mode for deterministic runtime env precedence.
   - Added stronger auth-deny debug logging and safer truthy parsing for allow-all toggles.
   - `tools/terminal_tool.py` now rejects invalid relative/host-only `TERMINAL_CWD` values for containerized backends.

4. Agent quality-of-life and model defaults
   - Updated MoA reference model from `openai/gpt-5.2-pro` to `openai/gpt-5.2`.
   - Scratchpad tags are normalized with think-block handling for visibility filtering.

5. Tests added/expanded
   - New tests: `tests/gateway/test_discord_archive.py`, `tests/gateway/test_channel_directory.py`, `tests/gateway/test_run_reset.py`.
   - Expanded tests for Discord gateway config/context behavior, session transcript fallback, multimodal persistence, and agent multimodal/scratchpad behavior.

