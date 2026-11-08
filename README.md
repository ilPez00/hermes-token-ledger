# hermes-token-ledger

Track LLM token usage and estimated cost per session in Hermes Agent.

## What it does

- Accumulates `prompt_tokens` / `output_tokens` / cache / reasoning tokens from every
  `post_api_request` hook into a SQLite ledger under the plugin data dir.
- Estimates USD cost using a small built-in pricing table (OpenAI, Anthropic,
  DeepSeek, Google, Meta, Qwen).
- Exposes `token_report` and `token_reset` tools.
- Adds a `/tokens` slash command.
- Optional per-session budget with warnings via log.

## Install

```
hermes plugins install hermes-token-ledger
```

(once listed in the catalog)

## Config

```yaml
plugins:
  entries:
    hermes-token-ledger:
      settings:
        budget_tokens: 100000
        warn_threshold: 0.8
        estimate_cost: true
```

## Tools

- `token_report` — aggregate by session/all/today.
- `token_reset` — delete entries.

## License

MIT
EOF