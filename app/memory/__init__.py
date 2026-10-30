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

"""Memory and session management package for the Retail Returns Concierge."""

from app.memory.compactor import (
    DEFAULT_MAX_TOKENS_THRESHOLD,
    DEFAULT_MAX_TURNS_THRESHOLD,
    DEFAULT_RETAIN_RECENT_TURNS,
    HistoryCompactor,
    async_compact_conversation_history,
    compact_conversation_history,
    estimate_token_count,
)
from app.memory.session_store import (
    DEFAULT_JSON_PATH,
    DEFAULT_SQLITE_PATH,
    DEFAULT_STORAGE_DIR,
    PersistentSessionStore,
)

__all__ = [
    "DEFAULT_JSON_PATH",
    "DEFAULT_MAX_TOKENS_THRESHOLD",
    "DEFAULT_MAX_TURNS_THRESHOLD",
    "DEFAULT_RETAIN_RECENT_TURNS",
    "DEFAULT_SQLITE_PATH",
    "DEFAULT_STORAGE_DIR",
    "HistoryCompactor",
    "PersistentSessionStore",
    "async_compact_conversation_history",
    "compact_conversation_history",
    "estimate_token_count",
]
