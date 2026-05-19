#!/usr/bin/env bash
# Bridge: dispatch a git-evolve text subagent through the `codex` CLI.
#
# Contract:
#   - Reads prompt from --prompt-file.
#   - Optionally prepends agents/<name>.md with --inline-spec.
#   - Runs `codex exec` non-interactively with a frontier OpenAI model.
#   - Writes the final text output to --out.
#
# Used by /brainstorm for a Codex-backed persona slot. The command runs
# read-only with approvals disabled because brainstorm personas return text and
# must not mutate the campaign checkout.

set -euo pipefail

AGENT=""
MODEL="gpt-5.5"
REASONING_EFFORT="xhigh"
PROMPT_FILE=""
OUT=""
CWD=""
INLINE_SPEC=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent)       AGENT="$2"; shift 2 ;;
    --model)       MODEL="$2"; shift 2 ;;
    --effort)      REASONING_EFFORT="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --out)         OUT="$2"; shift 2 ;;
    --cwd)         CWD="$2"; shift 2 ;;
    --inline-spec) INLINE_SPEC=true; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$AGENT" || -z "$PROMPT_FILE" || -z "$OUT" ]]; then
  echo "usage: $0 --agent NAME --prompt-file PATH --out PATH [--model ID] [--effort low|medium|high|xhigh] [--cwd DIR] [--inline-spec]" >&2
  exit 2
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "prompt file not found: $PROMPT_FILE" >&2
  exit 2
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not on PATH - install Codex or run /codex:setup from openai/codex-plugin-cc" >&2
  exit 3
fi

mkdir -p "$(dirname "$OUT")"

if [[ "$INLINE_SPEC" == true ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  SPEC_FILE="$REPO_ROOT/agents/${AGENT}.md"
  if [[ -f "$SPEC_FILE" ]]; then
    COMBINED="$(mktemp)"
    {
      echo "# Agent Spec: ${AGENT}"
      echo "# (inlined by dispatch-codex-subagent.sh --inline-spec)"
      echo
      cat "$SPEC_FILE"
      echo
      echo "---"
      echo "# Task Prompt"
      echo
      cat "$PROMPT_FILE"
    } > "$COMBINED"
    PROMPT_FILE="$COMBINED"
    trap "rm -f '$COMBINED'" EXIT
  else
    echo "warning: --inline-spec set but agents/${AGENT}.md not found at $SPEC_FILE" >&2
  fi
fi

ARGS=(
  exec
  --model "$MODEL"
  --config "model_reasoning_effort=\"$REASONING_EFFORT\""
  --sandbox read-only
  --ask-for-approval never
  --search
  --output-last-message "$OUT"
)

if [[ -n "$CWD" ]]; then
  ARGS+=(--cd "$CWD")
fi

# Prompt is piped on stdin to avoid argv length limits. Stdout can contain
# progress text; the final answer is read from --output-last-message.
codex "${ARGS[@]}" - < "$PROMPT_FILE" > "${OUT}.log"
