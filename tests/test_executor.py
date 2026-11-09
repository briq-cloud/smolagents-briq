"""Offline contracts: actual ipykernel and bridge, trusted test code only."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from smolagents import CodeAgent, Model, tool
from smolagents.default_tools import FinalAnswerTool
from smolagents.models import ChatMessage

from smolagents_briq import BriqExecutor


class LocalSandbox:
    def __init__(self, tmp_path):
        self.id = "bq_local_test"
        self.root = tmp_path
        self.paths = {}
        self.commands = []
        self.destroy_calls = 0
        self.destroy_failure = False
        self.fail_install = False

    def local(self, path):
        return str(self.root / path.lstrip("/"))

    def write_file(self, path, data):
        local = Path(self.local(path))
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        self.paths[path] = local

    def exec(self, command, *, timeout):
        self.commands.append(command)
        if command[1:3] == ["-m", "pip"]:
            return SimpleNamespace(
                stdout="",
                stderr="install failed" if self.fail_install else "",
                exit_code=1 if self.fail_install else 0,
                timed_out=False,
            )
        args = [sys.executable] + [
            self.local(x) if x.startswith("/tmp/") else x for x in command[1:]
        ]
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return SimpleNamespace(
            stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode, timed_out=False
        )

    def destroy(self):
        self.destroy_calls += 1
        if self.destroy_failure:
            raise RuntimeError("destroy failed")
        for pid_file in self.root.rglob("pid"):
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


class LocalClient:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.closed = False
        self.create_kwargs = None

    def create_sandbox(self, **kwargs):
        self.create_kwargs = kwargs
        return self.sandbox

    def close(self):
        self.closed = True


@pytest.fixture
def local(tmp_path):
    client = LocalClient(LocalSandbox(tmp_path))
    yield client
    client.sandbox.destroy_failure = False
    client.sandbox.destroy()


@pytest.fixture
def executor(local):
    with BriqExecutor(client=local) as executor:
        yield executor


def test_persistent_kernel_variables_logs_and_expression(executor):
    executor.send_variables({"items": [2, 3]})
    assert executor('total = sum(items)\nprint("first")').logs == "first\n"
    assert executor("total + 4").output == "9"
    assert not executor("total").is_final_answer


def test_final_answer_and_safe_types(executor):
    executor.send_tools({"final_answer": FinalAnswerTool()})
    result = executor('final_answer({"answer": (42, b"ok")})')
    assert result.output == {"answer": (42, b"ok")}
    assert result.is_final_answer


def test_python_error_preserves_kernel(executor):
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        executor("marker = 9\n1 / 0")
    assert executor("marker").output == "9"


def test_large_code_uses_file(executor, local):
    assert executor("#" + "x" * 12000 + "\n6 * 7").output == "42"
    assert all(len(json.dumps(command)) < 8192 for command in local.sandbox.commands)


def test_output_is_bounded(executor):
    result = executor('print("x" * 100000)')
    assert len(result.logs) <= 65536


def test_timeout_poison(executor):
    executor.execution_timeout = 0.1
    with pytest.raises(TimeoutError):
        executor("import time\ntime.sleep(10)")
    with pytest.raises(RuntimeError, match="unusable"):
        executor("1")


def test_cleanup_retry_and_borrowed_client(local):
    executor = BriqExecutor(client=local)
    local.sandbox.destroy_failure = True
    with pytest.raises(RuntimeError, match="destroy failed"):
        executor.cleanup()
    local.sandbox.destroy_failure = False
    executor.cleanup()
    executor.cleanup()
    assert local.sandbox.destroy_calls == 2
    assert not local.closed


def test_init_failure_cleanup(local):
    local.sandbox.fail_install = True
    with pytest.raises(RuntimeError, match="install failed"):
        BriqExecutor(client=local)
    assert local.sandbox.destroy_calls == 1


@pytest.mark.parametrize("timeout", [0, -1, 41, float("nan"), float("inf")])
def test_invalid_timeout_before_create(local, timeout):
    with pytest.raises(ValueError, match="execution_timeout"):
        BriqExecutor(client=local, execution_timeout=timeout)
    assert local.create_kwargs is None


@tool
def double(value: int) -> int:
    """Double a number.

    Args:
        value: Number to double.
    """
    return value * 2


class FixedModel(Model):
    def __init__(self):
        super().__init__()
        self.actions = iter(["answer = double(number)\nprint(answer)", "final_answer(answer)"])

    def generate(self, messages, **kwargs):
        return ChatMessage(role="assistant", content="<code>" + next(self.actions) + "</code>")


def test_released_codeagent_direct_injection(executor):
    agent = CodeAgent(
        tools=[double], model=FixedModel(), executor=executor, max_steps=2, verbosity_level=0
    )
    for _ in range(2):
        agent.model.actions = iter(
            ["answer = double(number)\nprint(answer)", "final_answer(answer)"]
        )
        assert agent.run("Double the number", additional_args={"number": 21}) == 42


@pytest.mark.parametrize("kind", ["ndarray", "PIL.Image"])
def test_rich_final_answer_rejected_before_decoder(executor, monkeypatch, kind):
    from smolagents.serialization import SafeSerializer

    def must_not_decode(_value):
        raise AssertionError("Rich object reached the host decoder")

    monkeypatch.setattr(SafeSerializer, "from_json_safe", must_not_decode)
    payload = {"nested": [{"__type__": kind, "data": ["x"], "dtype": "S1000000000"}]}
    with pytest.raises(ValueError, match="arrays as lists"):
        executor._deserialize_final_answer("safe:" + json.dumps(payload))


def test_guest_package_failure_propagates(executor):
    with pytest.raises(RuntimeError, match="CalledProcessError"):
        executor.install_packages(["--definitely-not-a-pip-option"])


@pytest.mark.parametrize("options", [{}, {"size": None}, {"size": "small"}])
def test_sdk_emitted_create_request(options):
    import httpx
    from briq_cloud import Briq, BriqError

    captured = []

    def receive(request):
        captured.append(request)
        return httpx.Response(400, json={"error": {"code": "invalid_argument", "message": "test"}})

    with Briq(
        api_key="test-key", base_url="http://localhost:3001", transport=httpx.MockTransport(receive)
    ) as client:
        with pytest.raises(BriqError):
            BriqExecutor(client=client, **options)
    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/briqs"
    expected = {
        "recipe": "sandbox",
        "wait": True,
        "waitTimeoutMs": 120000,
        "image": "python:3.12-slim",
        "sleep": "30m",
    }
    if options.get("size") is not None:
        expected["size"] = options["size"]
    assert json.loads(request.content) == expected
    assert request.headers.get("idempotency-key")


def test_owned_client_closed_on_setup_failure(local, monkeypatch):
    monkeypatch.setattr("smolagents_briq.executor.Briq", lambda: local)
    local.sandbox.fail_install = True
    with pytest.raises(RuntimeError):
        BriqExecutor()
    assert local.closed


def test_setup_and_cleanup_failures_both_visible(local):
    local.sandbox.fail_install = True
    local.sandbox.destroy_failure = True
    with pytest.raises(ExceptionGroup) as caught:
        BriqExecutor(client=local)
    assert len(caught.value.exceptions) == 2
    assert "install failed" in str(caught.value.exceptions[0])
    assert "destroy failed" in str(caught.value.exceptions[1])


def test_context_and_cleanup_failures_both_visible(local):
    with pytest.raises(ExceptionGroup) as caught:
        with BriqExecutor(client=local):
            local.sandbox.destroy_failure = True
            raise ValueError("body failed")
    assert [type(error) for error in caught.value.exceptions] == [ValueError, RuntimeError]


def test_explicit_size_image_and_sleep(local):
    with BriqExecutor(client=local, size="small", image="python:3.13-slim", sleep="3h"):
        assert local.create_kwargs == {"size": "small", "image": "python:3.13-slim", "sleep": "3h"}


def test_pickle_final_answer_rejected(executor):
    executor.send_tools({"final_answer": FinalAnswerTool()})
    with pytest.raises(ValueError, match="Only JSON"):
        executor(
            "class FinalAnswerException(BaseException): pass\n"
            "raise FinalAnswerException('pickle:AA==')"
        )


def test_final_answer_limit_is_explicit(executor):
    executor.send_tools({"final_answer": FinalAnswerTool()})
    with pytest.raises(RuntimeError, match="OutputLimitError"):
        executor("final_answer('x' * 70000)")


def test_remote_text_is_data(executor, tmp_path):
    sentinel = tmp_path / "host-must-not-execute"
    payload = f"__import__('pathlib').Path({str(sentinel)!r}).touch()"
    executor.send_tools({"final_answer": FinalAnswerTool()})
    assert executor(f"final_answer({payload!r})").output == payload
    assert not sentinel.exists()
