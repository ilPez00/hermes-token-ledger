"""hermes-token-ledger — per-session LLM token + cost ledger.

Accumulates usage from ``post_api_request`` into a SQLite ledger under the plugin
data dir, exposes ``token_report`` / ``token_reset`` tools and a ``/tokens``
slash command, and warns (via log) when a per-session budget is approached.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any, Optional

from plugins.plugin_storage import plugin_db

logger = logging.getLogger(__name__)

_PLUGIN_ID = "hermes-token-ledger"
_DB_FILE = "ledger.db"

# USD per 1M tokens (input, output). Unknown models return None.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4": (30.00, 60.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "llama-3.3-70b": (0.59, 0.79),
    "qwen": (0.30, 0.90),
}


_CTX = None


def _cfg(key: str, default: Any = None) -> Any:
    global _CTX
    if _CTX is not None:
        try:
            return _CTX.get_config(key, default)
        except Exception:
            pass
    # fallback (should not happen)
    return default


def _db() -> sqlite3.Connection:
    conn = plugin_db(_PLUGIN_ID, _DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            session_id TEXT,
            turn_id TEXT,
            api_request_id TEXT,
            model TEXT,
            provider TEXT,
            prompt_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read INTEGER DEFAULT 0,
            cache_write INTEGER DEFAULT 0,
            reasoning INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            api_duration REAL,
            finish_reason TEXT
        )"""
    )
    conn.commit()
    return conn


def _price_for(model: str) -> Optional[tuple[float, float]]:
    if not model:
        return None
    low = model.lower()
    # longest match first
    for name in sorted(_PRICES, key=len, reverse=True):
        if name in low:
            return _PRICES[name]
    return None


def _estimate_cost(model: str, prompt_tokens: int, output_tokens: int) -> Optional[float]:
    if not _cfg("estimate_cost", True):
        return None
    price = _price_for(model)
    if price is None:
        return None
    inp, out = price
    return round((prompt_tokens / 1_000_000.0) * inp + (output_tokens / 1_000_000.0) * out, 6)


def _session_tokens(session_id: str) -> int:
    if not session_id:
        return 0
    try:
        conn = _db()
        row = conn.execute(
            "SELECT COALESCE(SUM(total_tokens),0) AS t FROM calls WHERE session_id=?",
            (session_id,),
        ).fetchone()
        conn.close()
        return int(row["t"] or 0)
    except Exception:
        return 0


def _on_post_api_request(**kwargs: Any) -> None:
    try:
        usage = kwargs.get("usage") or {}
        if not isinstance(usage, dict) or not usage:
            return
        model = str(kwargs.get("model") or kwargs.get("response_model") or "")
        provider = str(kwargs.get("provider") or "")
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output = int(usage.get("output_tokens") or 0)
        cache_read = int(usage.get("cache_read_tokens") or 0)
        cache_write = int(usage.get("cache_write_tokens") or 0)
        reasoning = int(usage.get("reasoning_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + output))
        conn = _db()
        conn.execute(
            """INSERT INTO calls (ts, session_id, turn_id, api_request_id, model, provider,
               prompt_tokens, output_tokens, cache_read, cache_write, reasoning, total_tokens,
               api_duration, finish_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                float(kwargs.get("ended_at") or time.time()),
                str(kwargs.get("session_id") or ""),
                str(kwargs.get("turn_id") or ""),
                str(kwargs.get("api_request_id") or ""),
                model,
                provider,
                prompt,
                output,
                cache_read,
                cache_write,
                reasoning,
                total,
                float(kwargs.get("api_duration") or 0.0),
                str(kwargs.get("finish_reason") or ""),
            ),
        )
        conn.commit()
        conn.close()

        # budget warning
        budget = int(_cfg("budget_tokens", 0) or 0)
        if budget > 0:
            sid = str(kwargs.get("session_id") or "")
            used = _session_tokens(sid)
            threshold = float(_cfg("warn_threshold", 0.8) or 0.8)
            if used >= budget:
                logger.warning(
                    "hermes-token-ledger: session budget EXCEEDED (%d/%d tokens)", used, budget
                )
            elif used >= int(budget * threshold):
                logger.warning(
                    "hermes-token-ledger: approaching budget (%d/%d tokens, %.0f%%)",
                    used, budget, 100.0 * used / budget,
                )
    except Exception as exc:  # never break the agent loop
        logger.debug("hermes-token-ledger: post_api_request failed: %s", exc)


def _report(scope: str = "session", session_id: str = "", limit: int = 20) -> dict[str, Any]:
    conn = _db()
    where = ""
    params: list[Any] = []
    if scope == "session":
        where = "WHERE session_id=?"
        params = [session_id or ""]
    elif scope == "today":
        # last 24h
        where = "WHERE ts >= ?"
        params = [time.time() - 86400]
    rows = conn.execute(
        f"SELECT * FROM calls {where} ORDER BY id DESC LIMIT ?", (*params, max(1, min(int(limit), 200)))
    ).fetchall()
    agg = conn.execute(
        f"SELECT COUNT(*) AS n, COALESCE(SUM(prompt_tokens),0) AS p, "
        f"COALESCE(SUM(output_tokens),0) AS o, COALESCE(SUM(cache_read),0) AS cr, "
        f"COALESCE(SUM(cache_write),0) AS cw, COALESCE(SUM(reasoning),0) AS r, "
        f"COALESCE(SUM(total_tokens),0) AS t, COALESCE(SUM(api_duration),0) AS d "
        f"FROM calls {where}",
        tuple(params),
    ).fetchone()
    conn.close()

    # cost estimate across returned scope
    cost = 0.0
    have_cost = False
    # compute from all rows in scope (not just recent)
    conn2 = _db()
    all_rows = conn2.execute(
        f"SELECT model, prompt_tokens, output_tokens FROM calls {where}", tuple(params)
    ).fetchall()
    conn2.close()
    for r in all_rows:
        c = _estimate_cost(r["model"], int(r["prompt_tokens"] or 0), int(r["output_tokens"] or 0))
        if c is not None:
            cost += c
            have_cost = True

    recent = []
    for r in rows:
        recent.append({
            "ts": r["ts"],
            "session_id": r["session_id"],
            "model": r["model"],
            "provider": r["provider"],
            "prompt_tokens": r["prompt_tokens"],
            "output_tokens": r["output_tokens"],
            "total_tokens": r["total_tokens"],
            "finish_reason": r["finish_reason"],
            "cost_usd": _estimate_cost(r["model"], int(r["prompt_tokens"] or 0), int(r["output_tokens"] or 0)),
        })

    return {
        "scope": scope,
        "session_id": session_id if scope == "session" else None,
        "calls": int(agg["n"] or 0),
        "prompt_tokens": int(agg["p"] or 0),
        "output_tokens": int(agg["o"] or 0),
        "cache_read_tokens": int(agg["cr"] or 0),
        "cache_write_tokens": int(agg["cw"] or 0),
        "reasoning_tokens": int(agg["r"] or 0),
        "total_tokens": int(agg["t"] or 0),
        "api_duration_s": round(float(agg["d"] or 0), 3),
        "estimated_cost_usd": round(cost, 6) if have_cost else None,
        "budget_tokens": int(_cfg("budget_tokens", 0) or 0),
        "recent": recent,
    }


def _handle_token_report(args: Any = None, session_id: str = "", **_kw: Any) -> str:
    args = args if isinstance(args, dict) else {}
    scope = str(args.get("scope") or "session").lower()
    if scope not in {"session", "all", "today"}:
        scope = "session"
    limit = args.get("limit", 20)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 20
    report = _report(scope=scope, session_id=session_id, limit=limit)
    return json.dumps(report, indent=2)


def _handle_token_reset(args: Any = None, session_id: str = "", **_kw: Any) -> str:
    args = args if isinstance(args, dict) else {}
    scope = str(args.get("scope") or "session").lower()
    conn = _db()
    if scope == "all":
        conn.execute("DELETE FROM calls")
        conn.commit()
        conn.close()
        return json.dumps({"ok": True, "deleted": "all"})
    # session scope
    sid = session_id or str(args.get("session_id") or "")
    cur = conn.execute("DELETE FROM calls WHERE session_id=?", (sid,))
    conn.commit()
    n = cur.rowcount
    conn.close()
    return json.dumps({"ok": True, "deleted": n, "session_id": sid})


def _handle_slash(raw_args: str = "") -> Optional[str]:
    tokens = (raw_args or "").strip().split()
    sub = tokens[0].lower() if tokens else "status"
    if sub in {"", "status", "report"}:
        scope = "session"
        if len(tokens) > 1 and tokens[1].lower() in {"all", "today"}:
            scope = tokens[1].lower()
        # slash has no session_id; use global aggregation by treating session scope as all if no id
        # We'll report 'all' for slash to be useful.
        rep = _report(scope="all", session_id="", limit=10)
        return "Token ledger (all sessions):\n" + json.dumps(rep, indent=2)
    if sub == "reset":
        scope = tokens[1].lower() if len(tokens) > 1 else "session"
        if scope == "all":
            conn = _db()
            conn.execute("DELETE FROM calls")
            conn.commit()
            conn.close()
            return "Cleared all token ledger entries."
        return "Use /tokens reset all to clear everything (session scoping not available from slash)."
    if sub in {"help", "-h", "--help"}:
        return (
            "/tokens — LLM token + cost ledger\n"
            "  status [all|today]   Show totals (default all)\n"
            "  reset all            Clear all entries\n"
        )
    return f"Unknown subcommand: {sub}\nUse /tokens help"


def register(ctx) -> None:
    global _CTX
    _CTX = ctx
    ctx.register_tool(
        name="token_report",
        toolset="token_ledger",
        schema={
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["session", "all", "today"],
                    "description": "Which calls to aggregate. Default session.",
                },
                "limit": {
                    "type": "integer",
                    "description": "How many recent calls to include. Default 20.",
                },
            },
        },
        handler=_handle_token_report,
        description="Report LLM token usage and estimated cost from the ledger.",
        emoji="📊",
    )
    ctx.register_tool(
        name="token_reset",
        toolset="token_ledger",
        schema={
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["session", "all"],
                    "description": "Which entries to delete. Default session.",
                },
            },
        },
        handler=_handle_token_reset,
        description="Reset token ledger entries (session or all).",
        emoji="🧹",
    )
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_command("tokens", handler=_handle_slash, description="Token + cost ledger.")
