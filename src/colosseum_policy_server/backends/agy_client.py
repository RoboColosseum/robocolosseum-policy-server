"""Gemini through the official Antigravity CLI (``agy``), billed to the Client's Gemini API key.

Selected by ``llm_endpoint: antigravity`` on a ``gemini-*`` model. It replaces the Inspect agent's
HTTP client and keeps the same contract: one ``complete()`` call returns one proposed robot-tool
call, which the agent validates and executes as before.

Each call runs ``agy -p`` once, in a fresh temporary workspace:

- The agent's conversation (system prompt, history, tool results) becomes one text prompt.
- Camera frames are written into the workspace as image files and referenced with ``@path``;
  ``agy`` attaches them for the model. Its stdin stream-json input accepts text blocks only.
- ``--json-schema`` constrains the answer to exactly one of the embodiment's robot tools.
- ``--settings {"modelProvider": "gemini"}`` selects API-key auth for this call only, and
  ``GEMINI_API_KEY`` is set in the child environment alone. The host's agy settings and
  sign-in are never read or changed.
- Headless ``agy`` denies its own tools that need approval (shell, edits), so the model can only
  answer. The workspace holds nothing but this call's images.

A command-line argument is limited to 128 KiB on Linux, and ``agy -p`` does not read the prompt
from stdin. When a long trial's history no longer fits, the oldest turns after the task message
are left out of that call, with a note saying how many.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from inspect_robots_agent._llm import AssistantMessage, ToolCall

PROMPT_LIMIT_BYTES = 120_000  # below Linux MAX_ARG_STRLEN (131072) for the -p argument
EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
INSTALL_HINT = 'curl -fsSL https://antigravity.google/cli/install.sh | bash'
_IMAGE_URL = re.compile(r'data:image/(png|jpeg|gif|webp);base64,([A-Za-z0-9+/=]+)')
_SETTINGS = json.dumps({'modelProvider': 'gemini'})
_ENDPOINT = 'antigravity-cli://local/print'
_INSTRUCTIONS = (
    '\n\nYou control a physical robot through the robot tools listed below. Return exactly one '
    'next robot-tool proposal as JSON matching the supplied schema: "content" is a short note, '
    '"tool_calls" holds one {"name", "arguments"} object. These robot tools are not your own CLI '
    'tools: the robot runtime validates and executes your proposal and returns its result with '
    'the next observation. Do not run shell commands, edit files or browse; camera images are '
    'attached with @ references.'
)


class AgyClient:
    """Blocking ``agy -p`` adapter with the Inspect agent's ``complete()`` interface."""

    def __init__(self, model: str, *, api_key: str, command: str | None = None,
                 timeout_s: float = 300.0, attempts: int = 2, capture: Any = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError('Antigravity needs the Client\'s Gemini API key')
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError('agy_timeout_s must be finite and > 0')
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError('agy_attempts must be an integer >= 1')
        executable = shutil.which(command or 'agy')
        if executable is None and command is None:
            fallback = Path.home() / '.local' / 'bin' / 'agy'  # the installer's default location
            executable = str(fallback) if os.access(fallback, os.X_OK) else None
        if executable is None:
            raise ValueError(f'Antigravity CLI (agy) not found. Install it with: {INSTALL_HINT} '
                             '(or set agy_command to its path)')
        self.command = str(Path(executable).resolve())
        self.model = model
        self.timeout_s = float(timeout_s)
        self.attempts = attempts
        self._env = {**os.environ, 'GEMINI_API_KEY': api_key.strip()}
        self._capture = capture

    def slug(self, effort: str | float | None) -> str:
        """agy names a Gemini model and its thinking level together: gemini-4-hydrogen-medium."""
        if self.model.rsplit('-', 1)[-1] in EFFORTS:
            return self.model
        if effort is None:
            return self.model
        if effort not in EFFORTS:
            raise ValueError(f'Antigravity effort must be one of {", ".join(EFFORTS)}')
        return f'{self.model}-{effort}'

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                 temperature: float | None = None,
                 reasoning_effort: str | float | None = None) -> AssistantMessage:
        """Return one proposed robot-tool call, including token usage."""
        if temperature is not None:
            raise ValueError('temperature is not supported by the Antigravity CLI')
        slug = self.slug(reasoning_effort)
        schema = decision_schema(tools)
        last_error = 'no attempt made'
        for attempt in range(self.attempts):
            envelope, error = self._run(messages, tools, schema, slug, attempt)
            if envelope is not None:
                try:
                    return parse_decision(envelope, tools)
                except RuntimeError as exc:
                    # The model can answer without the schema, e.g. after trying a denied tool.
                    last_error = str(exc)
                    continue
            last_error = error or last_error
        raise RuntimeError(f'Antigravity CLI failed after {self.attempts} attempt(s): {last_error}')

    def _run(self, messages, tools, schema, slug, attempt):
        start = time.time()
        result = None
        envelope = None
        error = None
        try:
            with TemporaryDirectory(prefix='colosseum-agy-') as directory:
                workspace = Path(directory)
                prompt = build_prompt(messages, tools, workspace)
                schema_path = workspace / 'decision.schema.json'
                schema_path.write_text(json.dumps(schema))
                command = [self.command, '-p', prompt, '--model', slug, '--output-format', 'json',
                           '--json-schema', str(schema_path), '--settings', _SETTINGS,
                           '--disable-slash-commands']
                result = subprocess.run(command, cwd=workspace, env=self._env, capture_output=True,
                                        text=True, timeout=self.timeout_s, stdin=subprocess.DEVNULL,
                                        check=False)
            try:
                envelope = json.loads(result.stdout)
            except ValueError:
                envelope = None
            if not isinstance(envelope, dict):
                envelope = None
                error = (f'agy exited {result.returncode} without a JSON result: '
                         f'{(result.stderr or result.stdout).strip()[:1000]}')
            elif result.returncode != 0 or envelope.get('status') != 'SUCCESS':
                error = (f"agy status {envelope.get('status')!r} (exit {result.returncode}): "
                         f"{str(envelope.get('response') or envelope.get('error') or result.stderr)[:1000]}")
                envelope = None
            return envelope, error
        except subprocess.TimeoutExpired:
            error = f'agy timed out after {self.timeout_s:g} s'
            return None, error
        except OSError as exc:
            error = f'could not run agy: {exc}'
            return None, error
        finally:
            if self._capture is not None:
                self._capture.record(
                    attempt=attempt, endpoint=_ENDPOINT,
                    request={'model': slug, 'messages': messages, 'tools': tools,
                             'output_schema': schema},
                    status=result.returncode if result is not None else None,
                    response_text=result.stdout if result is not None else None,
                    error=error, t_start=start, duration_s=time.time() - start)

    def close(self) -> None:
        """Nothing to release: each call's process and workspace end with the call."""


def build_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]], workspace: Path) -> str:
    """One text prompt: system text, robot tools, then the history with @image references."""
    system = [str(m.get('content') or '') for m in messages if m.get('role') == 'system']
    tool_list = [{'name': t['function']['name'], 'description': t['function'].get('description', ''),
                  'parameters': t['function'].get('parameters', {})} for t in tools]
    head = ('\n\n'.join(system) + _INSTRUCTIONS + '\n\nRobot tools:\n' + json.dumps(tool_list, indent=1)
            + '\n\nConversation history in order; the last message is the current state:')
    turns = []
    images = 0
    for message in messages:
        role = message.get('role')
        if role == 'system':
            continue
        lines = [f'\n[{role}]']
        if message.get('tool_call_id'):
            lines.append(f"Tool result for {message['tool_call_id']}:")
        content = message.get('content')
        parts = [{'type': 'text', 'text': content}] if isinstance(content, str) else (content or [])
        for part in parts:
            if part.get('type') == 'text':
                lines.append(str(part.get('text', '')))
            elif part.get('type') == 'image_url':
                match = _IMAGE_URL.fullmatch(part['image_url']['url'])
                if match is None:
                    raise RuntimeError('Antigravity needs inline PNG/JPEG/GIF/WebP camera images')
                images += 1
                extension = 'jpg' if match[1] == 'jpeg' else match[1]
                path = workspace / f'image_{images:03d}.{extension}'
                path.write_bytes(base64.b64decode(match[2]))
                lines.append(f'(image {images}: @{path} )')
            else:
                raise RuntimeError(f"Unsupported content part for Antigravity: {part.get('type')}")
        if message.get('tool_calls'):
            calls = [{'name': c['function']['name'], 'arguments': c['function']['arguments']}
                     for c in message['tool_calls']]
            lines.append('Tool call: ' + json.dumps(calls))
        turns.append('\n'.join(lines))
    return fit(head, turns)


def fit(head: str, turns: list[str], limit: int = PROMPT_LIMIT_BYTES) -> str:
    """Keep the task message and the newest turns that fit in one command-line argument."""
    size = lambda text: len(text.encode())  # noqa: E731
    full = head + ''.join(turns)
    if size(full) <= limit:
        return full
    first, rest = (turns[0], turns[1:]) if turns else ('', [])
    kept = []
    budget = limit - size(head) - size(first) - 200  # room for the omission note
    for turn in reversed(rest):
        if size(turn) > budget:
            break
        kept.insert(0, turn)
        budget -= size(turn)
    if not kept and rest:
        raise RuntimeError('The current observation alone exceeds the agy prompt limit')
    omitted = len(rest) - len(kept)
    prompt = head + first + f'\n\n[{omitted} earlier message(s) omitted to fit the CLI prompt limit]' + ''.join(kept)
    if size(prompt) > limit:
        raise RuntimeError('The task message and current observation exceed the agy prompt limit')
    return prompt


def decision_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Constrain the answer to one call of the embodiment's robot tools."""
    choices = [{'type': 'object', 'additionalProperties': False,
                'description': t['function'].get('description', ''),
                'properties': {'name': {'const': t['function']['name'], 'type': 'string'},
                               'arguments': t['function']['parameters']},
                'required': ['name', 'arguments']} for t in tools]
    if not choices:
        raise RuntimeError('Antigravity needs a bound robot toolset')
    return {'type': 'object', 'additionalProperties': False,
            'properties': {'content': {'type': 'string'},
                           'tool_calls': {'type': 'array', 'minItems': 1, 'maxItems': 1,
                                          'items': {'anyOf': choices}}},
            'required': ['content', 'tool_calls']}


def parse_decision(envelope: dict[str, Any], tools: list[dict[str, Any]]) -> AssistantMessage:
    decision = envelope.get('structured_output')
    if not isinstance(decision, dict):
        denied = envelope.get('denied_actions')
        raise RuntimeError('agy returned no structured_output'
                           + (f' (denied tools: {denied})' if denied else '')
                           + f": {str(envelope.get('response', ''))[:300]}")
    content, calls = decision.get('content'), decision.get('tool_calls')
    if not isinstance(content, str) or not isinstance(calls, list) or len(calls) != 1:
        raise RuntimeError('agy must return text and exactly one robot-tool proposal')
    call = calls[0]
    if not isinstance(call, dict) or call.get('name') not in {t['function']['name'] for t in tools}:
        raise RuntimeError('agy returned an unknown robot tool')
    arguments = call.get('arguments')
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = None
    if not isinstance(arguments, dict):
        raise RuntimeError('agy robot-tool arguments must be a JSON object')
    usage = envelope.get('usage') if isinstance(envelope.get('usage'), dict) else {}
    names = {'input_tokens': 'input_tokens', 'output_tokens': 'output_tokens',
             'cache_read_tokens': 'cache_read_input_tokens', 'thinking_tokens': 'reasoning_tokens'}
    counts = {names[k]: v for k, v in usage.items()
              if k in names and isinstance(v, int) and not isinstance(v, bool)}
    return AssistantMessage(content=content or None,
                            tool_calls=(ToolCall(id=f'agy_{uuid.uuid4().hex}', name=call['name'],
                                                 arguments=json.dumps(arguments)),),
                            usage=counts or None)
