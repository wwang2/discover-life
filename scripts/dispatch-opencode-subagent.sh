#!/usr/bin/env bash
# Bridge: dispatch a research-everything subagent via the `opencode` CLI.
#
# Used by the Claude-Code-host orchestrator to invoke a subagent through
# OpenCode (typically routed to OpenRouter). The OpenCode-host orchestrator
# does NOT use this bridge — it calls OpenCode's native task tool directly.
#
# Contract:
#   - Reads prompt from --prompt-file (path to a file with the full prompt).
#   - Runs `opencode run` non-interactively with the requested agent + model.
#   - Writes the agent's final text output to --out.
#   - Exits 0 on success, non-zero on failure. Failure output goes to stderr.
#
# The orchestrator reads --out the same way it would process a Task() result.
#
# Usage:
#   scripts/dispatch-opencode-subagent.sh \
#     --agent verifier \
#     --model openrouter/anthropic/claude-haiku-4.5 \
#     --prompt-file /tmp/verifier-prompt.txt \
#     --out /tmp/verifier-result.txt \
#     [--cwd .worktrees/orbit-name]

set -euo pipefail

AGENT=""
MODEL=""
PROMPT_FILE=""
OUT=""
CWD=""
INLINE_SPEC=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent)       AGENT="$2"; shift 2 ;;
    --model)       MODEL="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --out)         OUT="$2"; shift 2 ;;
    --cwd)         CWD="$2"; shift 2 ;;
    --inline-spec) INLINE_SPEC=true; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$AGENT" || -z "$PROMPT_FILE" || -z "$OUT" ]]; then
  echo "usage: $0 --agent NAME --prompt-file PATH --out PATH [--model ID] [--cwd DIR]" >&2
  exit 2
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "prompt file not found: $PROMPT_FILE" >&2
  exit 2
fi

if ! command -v opencode >/dev/null 2>&1; then
  echo "opencode CLI not on PATH — install from https://opencode.ai" >&2
  exit 3
fi

if [[ -n "$MODEL" && -z "${OPENROUTER_API_KEY:-}" && "$MODEL" == openrouter/* ]]; then
  echo "OPENROUTER_API_KEY is not set but --model requests $MODEL" >&2
  exit 4
fi

mkdir -p "$(dirname "$OUT")"

# If --inline-spec is set, prepend the agent's full .md spec to the prompt
# file. This ensures the agent has its complete instructions even when running
# remotely via OpenRouter (where it cannot read the repo filesystem).
if [[ "$INLINE_SPEC" == true ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  SPEC_FILE="$REPO_ROOT/agents/${AGENT}.md"
  if [[ -f "$SPEC_FILE" ]]; then
    COMBINED="$(mktemp)"
    {
      echo "# Agent Spec: ${AGENT}"
      echo "# (inlined by dispatch-opencode-subagent.sh --inline-spec)"
      echo
      cat "$SPEC_FILE"
      echo
      echo "---"
      echo "# Task Prompt"
      echo
      cat "$PROMPT_FILE"
    } > "$COMBINED"
    PROMPT_FILE="$COMBINED"
    # Clean up temp file on exit
    trap "rm -f '$COMBINED'" EXIT
  else
    echo "warning: --inline-spec set but agents/${AGENT}.md not found at $SPEC_FILE" >&2
  fi
fi

# Build argv. `opencode run` is the non-interactive headless mode; --agent
# selects a subagent definition from .opencode/agents/, --model overrides the
# model for this invocation. We pipe the prompt on stdin rather than via a
# shell arg to avoid argv length limits.
ARGS=(run --agent "$AGENT")
if [[ -n "$MODEL" ]]; then
  ARGS+=(--model "$MODEL")
fi

if [[ -n "$CWD" ]]; then
  (cd "$CWD" && opencode "${ARGS[@]}" < "$PROMPT_FILE") > "$OUT"
else
  opencode "${ARGS[@]}" < "$PROMPT_FILE" > "$OUT"
fi
