"""Shared test fixtures."""
import json
import sys

import pytest

# A scripted stand-in for the Antigravity CLI. It records each call (arguments, prompt, the
# @-referenced images, its workspace and GEMINI_API_KEY) and prints the next scripted result.
FAKE_AGY = r'''
import json, os, re, sys
argv = sys.argv[1:]
prompt = argv[argv.index('-p') + 1]
images = re.findall(r'@(\S+)', prompt)
seen = [{'path': p, 'png': open(p, 'rb').read(8) == b'\x89PNG\r\n\x1a\n'} for p in images]
log = os.environ['FAKE_AGY_LOG']
calls = json.load(open(log)) if os.path.exists(log) else []
calls.append({'argv': argv, 'prompt': prompt, 'images': seen, 'cwd': os.getcwd(),
              'key': os.environ.get('GEMINI_API_KEY'),
              'schema': json.load(open(argv[argv.index('--json-schema') + 1]))})
json.dump(calls, open(log, 'w'))
replies = json.loads(os.environ['FAKE_AGY_REPLIES'])
print(json.dumps(replies[min(len(calls), len(replies)) - 1]))
'''

AGY_MOVE = {'content': 'Move both arms', 'tool_calls': [{'name': 'move_joints', 'arguments': {
    'targets': {'left_joint1': .03, 'right_joint1': -.03}, 'note': 'Move both arms'}}]}


def agy_envelope(decision=AGY_MOVE, **extra):
    return {'status': 'SUCCESS', 'response': json.dumps(decision), 'structured_output': decision,
            'num_turns': 1, 'usage': {'input_tokens': 9000, 'output_tokens': 50,
                                      'cache_read_tokens': 4000, 'thinking_tokens': 30}, **extra}


@pytest.fixture
def fake_agy(tmp_path, monkeypatch):
    """Put a fake `agy` first on PATH. Returns (script, calls(), replies(*envelopes))."""
    bin_dir = tmp_path / 'fake-bin'
    bin_dir.mkdir()
    script = bin_dir / 'agy'
    script.write_text(f'#!{sys.executable}\n{FAKE_AGY}')
    script.chmod(0o755)
    log = tmp_path / 'agy-calls.json'
    monkeypatch.setenv('PATH', f"{bin_dir}:{__import__('os').environ.get('PATH', '')}")
    monkeypatch.setenv('FAKE_AGY_LOG', str(log))
    monkeypatch.setenv('GEMINI_API_KEY', 'host-key-must-not-be-used')

    def replies(*items):
        monkeypatch.setenv('FAKE_AGY_REPLIES', json.dumps(list(items)))
    replies(agy_envelope())
    return script, (lambda: json.loads(log.read_text()) if log.exists() else []), replies
