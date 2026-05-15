"""
Claude Code Subagent Backend

Replaces LiteLLM API calls with `claude -p` subprocess calls so that all LLM
inference runs through the user's Claude Code OAuth subscription rather than
billing a separate API key.

Auth note: do NOT pass --bare; the subprocess inherits the shell environment
and keychain so OAuth credentials are used automatically.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Optional, Type, Union

from pydantic import BaseModel

from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import APIBackend

_NVM_CLAUDE = "/home/intern2/.nvm/versions/node/v24.15.0/bin/claude"
CLAUDE_BINARY: str = os.environ.get("CLAUDE_BINARY") or shutil.which("claude") or _NVM_CLAUDE

_MODEL = os.environ.get("CLAUDE_CODE_MODEL", "claude-sonnet-4-6")
_TIMEOUT = int(os.environ.get("CLAUDE_CODE_TIMEOUT", "300"))


class ClaudeCodeSubagentBackend(APIBackend):
    """APIBackend that delegates every chat completion to a `claude -p` subprocess."""

    def _create_chat_completion_inner_function(
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, str | None]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_parts = [m["content"] for m in messages if m["role"] != "system"]

        system = "\n".join(system_parts)
        prompt = "\n".join(user_parts)

        # JSON mode: inject instruction into system prompt; JSONParser in base.py handles parsing
        if response_format:
            system = (system + "\nRespond with valid JSON only.").strip()

        # Pass prompt via stdin to avoid CLI parsing errors when prompt starts with dashes.
        # --print with capture_output=True (non-TTY stdout) enables non-interactive mode.
        cmd: list[str] = [
            CLAUDE_BINARY, "--print",
            "--output-format", "json",
            "--no-session-persistence",
            "--model", _MODEL,
        ]
        if system:
            cmd.extend(["--system-prompt", system])

        logger.info(f"[ClaudeCodeSubagent] calling claude subprocess (model={_MODEL})", tag="llm_messages")

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"claude subprocess exited {result.returncode}: {result.stderr[:500]}"
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"claude subprocess returned non-JSON stdout: {result.stdout[:300]}") from exc

        if data.get("is_error"):
            raise RuntimeError(f"claude returned error: {data.get('result', '')[:300]}")

        content: str = data.get("result", "")
        logger.info(f"[ClaudeCodeSubagent] done (cost_usd={data.get('total_cost_usd')}, duration_ms={data.get('duration_ms')})", tag="token_cost")
        return content, "stop"

    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        # Claude Code does not provide an embedding endpoint; delegate to LiteLLM.
        from rdagent.oai.backend.litellm import LiteLLMAPIBackend
        return LiteLLMAPIBackend()._create_embedding_inner_function(input_content_list)

    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        return len(str(messages)) // 4

    def supports_response_schema(self) -> bool:
        return False

    @property
    def chat_token_limit(self) -> int:
        # claude-sonnet-4-6: 200k context window; reserve 16k for output
        return 184_000
