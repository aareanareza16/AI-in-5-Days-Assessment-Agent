# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistent session state store with SQLite/JSON disk persistence and async tasks.

Implements Requirement 3 & Requirement 4 of Category 2: Context & Memory:
- Persists session state, conversation turns, and compacted history summaries to
  disk using SQLite (`.db` / `.sqlite`) and/or JSON (`.json`) so state survives
  across conversation turns and process reloads.
- Provides asynchronous (`asyncio`) background saving and compaction tasks so disk
  I/O and summarization never block real-time chat latency.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.memory.compactor import HistoryCompactor
from app.observability.logger import get_structured_logger
from app.observability.pii_scrubber import scrub_pii, scrub_text

_store_logger = get_structured_logger("retail_returns_concierge.session_store")

DEFAULT_STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / ".session_store"
DEFAULT_SQLITE_PATH = DEFAULT_STORAGE_DIR / "retail_concierge_sessions.db"
DEFAULT_JSON_PATH = DEFAULT_STORAGE_DIR / "retail_concierge_sessions.json"


def _json_default(obj: Any) -> Any:
    """Fallback JSON serializer for Pydantic models, dates, sets, or custom objects."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


class PersistentSessionStore:
    """Disk-backed session store supporting SQLite + JSON persistence and async operations.

    Persists session `state`, `turns`, and `compacted_summary` across reloads.
    Supports scheduling non-blocking background save and compaction operations
    via `asyncio.create_task` and `asyncio.to_thread`.
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        json_mirror_path: str | Path | None = None,
        compactor: HistoryCompactor | None = None,
    ) -> None:
        """Initializes the persistent session store and ensures disk tables/files exist.

        Args:
            storage_path: Path to the SQLite database file (or `.json` file if
                JSON-only mode is desired). Defaults to `.session_store/retail_concierge_sessions.db`.
            json_mirror_path: Optional path to a JSON snapshot file updated alongside
                SQLite writes. Defaults to `.session_store/retail_concierge_sessions.json`
                when using the default storage path.
            compactor: Optional `HistoryCompactor` used during background save-and-compact.
        """
        resolved_path = (
            Path(storage_path) if storage_path is not None else DEFAULT_SQLITE_PATH
        )
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path: Path = resolved_path
        self.use_json_primary: bool = resolved_path.suffix.lower() == ".json"

        if json_mirror_path is not None:
            self.json_mirror_path: Path | None = Path(json_mirror_path)
            self.json_mirror_path.parent.mkdir(parents=True, exist_ok=True)
        elif storage_path is None:
            self.json_mirror_path = DEFAULT_JSON_PATH
        else:
            self.json_mirror_path = (
                resolved_path
                if self.use_json_primary
                else resolved_path.with_suffix(".json")
            )

        self.compactor: HistoryCompactor = compactor or HistoryCompactor()
        self._lock = threading.RLock()
        self._background_tasks: set[asyncio.Task[Any]] = set()

        if not self.use_json_primary:
            self._init_sqlite()
        else:
            self._init_json_file(self.storage_path)

    def _init_sqlite(self) -> None:
        """Creates the SQLite sessions table if it does not already exist."""
        with self._lock, sqlite3.connect(self.storage_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    app_name TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    turns_json TEXT NOT NULL,
                    compacted_summary TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _init_json_file(self, path: Path) -> None:
        """Initializes an empty JSON session dictionary file if missing."""
        with self._lock:
            if not path.exists():
                path.write_text("{}", encoding="utf-8")

    def _write_json_mirror(self, session_record: dict[str, Any]) -> None:
        """Updates the JSON snapshot file with `session_record`."""
        target = self.storage_path if self.use_json_primary else self.json_mirror_path
        if target is None:
            return
        with self._lock:
            data: dict[str, Any] = {}
            if target.exists():
                try:
                    raw = target.read_text(encoding="utf-8").strip()
                    if raw:
                        data = json.loads(raw)
                except (json.JSONDecodeError, OSError):
                    data = {}
            data[session_record["session_id"]] = session_record
            target.write_text(
                json.dumps(data, indent=2, default=_json_default),
                encoding="utf-8",
            )

    def save_session(
        self,
        session_id: str,
        state: dict[str, Any],
        turns: list[dict[str, Any]] | None = None,
        compacted_summary: str | None = None,
        user_id: str = "default_user",
        app_name: str = "app",
    ) -> dict[str, Any]:
        """Synchronously persists a session's state, turns, and compacted summary to disk.

        Merges with any existing persisted record for `session_id` so unspecified
        fields (`turns` or `compacted_summary`) are preserved across incremental updates.

        Args:
            session_id: Unique session identifier.
            state: Dictionary of session state key-value pairs.
            turns: Optional list of conversation turn dictionaries.
            compacted_summary: Optional compacted summary of older turns.
            user_id: User identifier owning the session.
            app_name: Application name (default 'app').

        Returns:
            The persisted session record dictionary.
        """
        with self._lock:
            existing = self.load_session(session_id)
            merged_state: dict[str, Any] = {}
            if existing and isinstance(existing.get("state"), dict):
                merged_state.update(existing["state"])
            merged_state.update(state)

            resolved_turns = (
                turns
                if turns is not None
                else (existing.get("turns", []) if existing else [])
            )
            resolved_summary = (
                compacted_summary
                if compacted_summary is not None
                else (
                    merged_state.get("compacted_history_summary")
                    or (existing.get("compacted_summary", "") if existing else "")
                )
            )
            now_iso = datetime.datetime.now(datetime.UTC).isoformat()

            scrubbed_state = scrub_pii(merged_state)
            scrubbed_turns = scrub_pii(resolved_turns)
            scrubbed_summary = scrub_text(str(resolved_summary))
            if "compacted_history_summary" in scrubbed_state:
                scrubbed_state["compacted_history_summary"] = scrubbed_summary

            state_json = json.dumps(scrubbed_state, default=_json_default)
            turns_json = json.dumps(scrubbed_turns, default=_json_default)

            record = {
                "session_id": session_id,
                "user_id": str(scrub_pii(user_id)),
                "app_name": app_name,
                "state": json.loads(state_json),
                "turns": json.loads(turns_json),
                "compacted_summary": scrubbed_summary,
                "updated_at": now_iso,
            }

            if not self.use_json_primary:
                with sqlite3.connect(self.storage_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO sessions (
                            session_id, user_id, app_name, state_json,
                            turns_json, compacted_summary, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            user_id = excluded.user_id,
                            app_name = excluded.app_name,
                            state_json = excluded.state_json,
                            turns_json = excluded.turns_json,
                            compacted_summary = excluded.compacted_summary,
                            updated_at = excluded.updated_at
                        """,
                        (
                            session_id,
                            record["user_id"],
                            app_name,
                            state_json,
                            turns_json,
                            scrubbed_summary,
                            now_iso,
                        ),
                    )
                    conn.commit()

            self._write_json_mirror(record)
            _store_logger.info(
                f"Persisted PII-scrubbed session '{session_id}' to disk.",
                session_id=session_id,
                event_type="SESSION_PERSISTENCE",
                context={
                    "turn_count": len(record["turns"]),
                    "has_compacted_summary": bool(scrubbed_summary),
                },
            )
            return record

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Synchronously loads a persisted session record from disk by `session_id`.

        Args:
            session_id: Unique session identifier to retrieve.

        Returns:
            Dictionary with `session_id`, `user_id`, `app_name`, `state`, `turns`,
            `compacted_summary`, and `updated_at`, or `None` if not found.
        """
        with self._lock:
            if not self.use_json_primary:
                with sqlite3.connect(self.storage_path) as conn:
                    row = conn.execute(
                        """
                        SELECT session_id, user_id, app_name, state_json,
                               turns_json, compacted_summary, updated_at
                        FROM sessions
                        WHERE session_id = ?
                        """,
                        (session_id,),
                    ).fetchone()
                if row is None:
                    return None
                return {
                    "session_id": row[0],
                    "user_id": row[1],
                    "app_name": row[2],
                    "state": json.loads(row[3]),
                    "turns": json.loads(row[4]),
                    "compacted_summary": row[5],
                    "updated_at": row[6],
                }

            if not self.storage_path.exists():
                return None
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                return data.get(session_id)
            except (json.JSONDecodeError, OSError):
                return None

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        state_updates: dict[str, Any] | None = None,
        auto_compact: bool = True,
    ) -> dict[str, Any]:
        """Appends a conversation turn to `session_id`, optionally compacting and saving."""
        with self._lock:
            existing = self.load_session(session_id) or {
                "session_id": session_id,
                "user_id": "default_user",
                "app_name": "app",
                "state": {},
                "turns": [],
                "compacted_summary": "",
            }
            turns = list(existing.get("turns", []))
            turns.append({"role": role, "content": content})

            summary = str(existing.get("compacted_summary") or "")
            merged_state = dict(existing.get("state", {}))
            if state_updates:
                merged_state.update(state_updates)

            if auto_compact and self.compactor.should_compact(turns):
                compaction_res = self.compactor.compact(turns, existing_summary=summary)
                turns = compaction_res["compacted_turns"]
                summary = compaction_res["summary"]
                merged_state["compacted_history_summary"] = summary
                merged_state["history_was_compacted"] = True

            return self.save_session(
                session_id=session_id,
                state=merged_state,
                turns=turns,
                compacted_summary=summary,
                user_id=existing.get("user_id", "default_user"),
                app_name=existing.get("app_name", "app"),
            )

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session from disk storage."""
        with self._lock:
            deleted = False
            if not self.use_json_primary:
                with sqlite3.connect(self.storage_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM sessions WHERE session_id = ?", (session_id,)
                    )
                    conn.commit()
                    deleted = cursor.rowcount > 0

            target = (
                self.storage_path if self.use_json_primary else self.json_mirror_path
            )
            if target and target.exists():
                try:
                    data = json.loads(target.read_text(encoding="utf-8"))
                    if session_id in data:
                        del data[session_id]
                        target.write_text(
                            json.dumps(data, indent=2, default=_json_default),
                            encoding="utf-8",
                        )
                        deleted = True
                except (json.JSONDecodeError, OSError):
                    pass
            return deleted

    def list_sessions(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Lists all persisted sessions, optionally filtered by `user_id`."""
        with self._lock:
            if not self.use_json_primary:
                with sqlite3.connect(self.storage_path) as conn:
                    if user_id:
                        rows = conn.execute(
                            "SELECT session_id FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
                            (user_id,),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT session_id FROM sessions ORDER BY updated_at DESC"
                        ).fetchall()
                results: list[dict[str, Any]] = []
                for (sid,) in rows:
                    loaded = self.load_session(sid)
                    if loaded is not None:
                        results.append(loaded)
                return results

            if not self.storage_path.exists():
                return []
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            records = list(data.values())
            if user_id:
                records = [r for r in records if r.get("user_id") == user_id]
            return records

    # ========================================================================
    # Asynchronous / Non-Blocking Memory Operations
    # ========================================================================

    async def save_session_async(
        self,
        session_id: str,
        state: dict[str, Any],
        turns: list[dict[str, Any]] | None = None,
        compacted_summary: str | None = None,
        user_id: str = "default_user",
        app_name: str = "app",
    ) -> dict[str, Any]:
        """Asynchronously saves session state and turns in a worker thread."""
        return await asyncio.to_thread(
            self.save_session,
            session_id,
            state,
            turns,
            compacted_summary,
            user_id,
            app_name,
        )

    async def load_session_async(self, session_id: str) -> dict[str, Any] | None:
        """Asynchronously loads a persisted session record from disk."""
        return await asyncio.to_thread(self.load_session, session_id)

    async def append_turn_async(
        self,
        session_id: str,
        role: str,
        content: str,
        state_updates: dict[str, Any] | None = None,
        auto_compact: bool = True,
    ) -> dict[str, Any]:
        """Asynchronously appends a turn, compacts if needed, and persists to disk."""
        return await asyncio.to_thread(
            self.append_turn,
            session_id,
            role,
            content,
            state_updates,
            auto_compact,
        )

    async def _background_save_and_compact_worker(
        self,
        session_id: str,
        state: dict[str, Any],
        turns: list[dict[str, Any]] | None,
        user_id: str,
        app_name: str,
    ) -> dict[str, Any]:
        """Internal coroutine that runs compaction and disk persistence off the main loop."""

        def _sync_work() -> dict[str, Any]:
            with self._lock:
                existing = self.load_session(session_id)
                active_turns = (
                    list(turns)
                    if turns is not None
                    else (list(existing.get("turns", [])) if existing else [])
                )
                existing_summary = str(
                    state.get("compacted_history_summary")
                    or (existing.get("compacted_summary", "") if existing else "")
                )
                merged_state = dict(state)
                if active_turns and self.compactor.should_compact(active_turns):
                    compaction = self.compactor.compact(
                        active_turns, existing_summary=existing_summary
                    )
                    active_turns = compaction["compacted_turns"]
                    existing_summary = compaction["summary"]
                    merged_state["compacted_history_summary"] = existing_summary
                    merged_state["history_was_compacted"] = True

                return self.save_session(
                    session_id=session_id,
                    state=merged_state,
                    turns=active_turns,
                    compacted_summary=existing_summary,
                    user_id=user_id,
                    app_name=app_name,
                )

        return await asyncio.to_thread(_sync_work)

    def schedule_background_save_and_compact(
        self,
        session_id: str,
        state: dict[str, Any],
        turns: list[dict[str, Any]] | None = None,
        user_id: str = "default_user",
        app_name: str = "app",
    ) -> asyncio.Task[dict[str, Any]] | None:
        """Schedules an asynchronous background task to compact and persist session state.

        When called inside a running `asyncio` event loop, launches a fire-and-forget
        `asyncio.Task` so chat response latency is never blocked by disk writes or
        history compaction. If no event loop is running, falls back to executing
        on a daemon background thread.

        Args:
            session_id: Unique session identifier.
            state: Current session state snapshot.
            turns: Optional conversation turns list.
            user_id: User identifier.
            app_name: Application identifier.

        Returns:
            The scheduled `asyncio.Task` if an event loop is running, else `None`.
        """
        state_copy = dict(state)
        turns_copy = list(turns) if turns is not None else None

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._background_save_and_compact_worker(
                    session_id=session_id,
                    state=state_copy,
                    turns=turns_copy,
                    user_id=user_id,
                    app_name=app_name,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return task
        except RuntimeError:
            # No running event loop; run non-blockingly on a daemon thread
            thread = threading.Thread(
                target=lambda: asyncio.run(
                    self._background_save_and_compact_worker(
                        session_id=session_id,
                        state=state_copy,
                        turns=turns_copy,
                        user_id=user_id,
                        app_name=app_name,
                    )
                ),
                daemon=True,
            )
            thread.start()
            return None

    async def flush_background_tasks(self) -> None:
        """Awaits all currently pending background persistence/compaction tasks."""
        if not self._background_tasks:
            return
        pending = list(self._background_tasks)
        await asyncio.gather(*pending, return_exceptions=True)
