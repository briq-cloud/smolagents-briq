"""Persistent smolagents execution through the briq Python SDK."""

from __future__ import annotations

import json
import math
import uuid
from importlib.resources import files
from threading import RLock
from typing import Any

from briq_cloud import Briq
from smolagents.local_python_executor import CodeOutput
from smolagents.monitoring import AgentLogger
from smolagents.remote_executors import RemotePythonExecutor


class BriqExecutor(RemotePythonExecutor):
    """One persistent Python kernel in an owned briq sandbox.

    Use as a context manager. An injected client stays open. Each executor owns
    and destroys its sandbox, including after a failed initialization. Calls are
    serialized; an uncertain execution outcome requires a new executor.
    """

    def __init__(
        self,
        additional_imports: list[str] | None = None,
        logger: AgentLogger | None = None,
        *,
        client: Briq | None = None,
        image: str = "python:3.12-slim",
        size: str | None = None,
        sleep: str = "30m",
        execution_timeout: float = 30,
    ) -> None:
        if not math.isfinite(execution_timeout) or not 0 < execution_timeout <= 40:
            raise ValueError("execution_timeout must be finite and in (0, 40]")
        super().__init__(additional_imports or [], logger or AgentLogger(), allow_pickle=False)
        self._lock = RLock()
        self._owned_client = client is None
        self._client = client if client is not None else Briq()
        self.sandbox = None
        self._closed = False
        self._poisoned = False
        self.execution_timeout = execution_timeout
        self._root = f"/tmp/smolagents-briq-{uuid.uuid4().hex}"
        try:
            self.sandbox = self._client.create_sandbox(image=image, size=size, sleep=sleep)
            self._check(
                self.sandbox.exec(
                    [
                        "python3",
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "smolagents==1.26.0",
                        "numpy>=1.26,<3",
                        "ipykernel>=6,<8",
                        "jupyter-client>=8,<9",
                    ],
                    timeout=60,
                )
            )
            self.sandbox.write_file(
                self._root + "/bridge.py",
                files("smolagents_briq").joinpath("_bridge.py").read_bytes(),
            )
            self._check(
                self.sandbox.exec(
                    ["python3", self._root + "/bridge.py", self._root, "boot"], timeout=25
                )
            )
            self.installed_packages = [
                "smolagents",
                "PIL",
                "pillow",
                "numpy",
                "ipykernel",
                "jupyter_client",
            ]
            self.installed_packages += self.install_packages(self.additional_imports)
        except BaseException as original:
            try:
                self.cleanup()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "Executor initialization and cleanup failed", [original, cleanup_error]
                ) from None
            raise

    @staticmethod
    def _check(result: Any) -> str:
        if result.timed_out:
            raise TimeoutError("briq command timed out")
        if result.exit_code != 0:
            raise RuntimeError("briq command failed: " + result.stderr[:4096])
        return result.stdout

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        with self._lock:
            if self._closed or self._poisoned:
                raise RuntimeError("Executor is closed or unusable; create a new executor")
            assert self.sandbox is not None
            path = self._root + "/action.py"
            self.sandbox.write_file(path, code.encode())
            try:
                raw = self._check(
                    self.sandbox.exec(
                        [
                            "python3",
                            self._root + "/bridge.py",
                            self._root,
                            "run",
                            path,
                            str(self.execution_timeout),
                        ],
                        timeout=self.execution_timeout + 15,
                    )
                )
                # JSON is the only host-side wire format. Never evaluate text/plain.
                if len(raw) > 1024 * 1024:
                    raise RuntimeError("Kernel response exceeds limit")
                result = json.loads(raw)
                if result["timed_out"]:
                    raise TimeoutError("Kernel execution timed out; executor is unusable")
                error = result["error"]
                logs = result["logs"]
                output = result["output"]
                if not isinstance(logs, str) or not (output is None or isinstance(output, str)):
                    raise ValueError("Malformed kernel response")
            except BaseException:
                self._poisoned = True
                raise
            if error:
                if not (
                    isinstance(error, dict)
                    and isinstance(error.get("name"), str)
                    and isinstance(error.get("value"), str)
                ):
                    self._poisoned = True
                    raise ValueError("Malformed kernel error")
                if error["name"] == self.FINAL_ANSWER_EXCEPTION:
                    value = error["value"]
                    try:
                        if not value.startswith("safe:"):
                            raise ValueError("Only JSON final answers are supported")
                        answer = self._deserialize_final_answer(value, allow_pickle=False)
                    except Exception:
                        self._poisoned = True
                        raise
                    return CodeOutput(
                        output=answer,
                        logs=logs,
                        is_final_answer=True,
                    )
                raise RuntimeError(f"{logs}\n{error['name']}: {error['value']}")
            return CodeOutput(output=output, logs=logs, is_final_answer=False)

    def install_packages(self, additional_imports: list[str]) -> list[str]:
        if additional_imports:
            # Specifications are argv values, never shell commands.
            self.run_code_raise_errors(
                "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', "
                + ", ".join(repr(item) for item in additional_imports)
                + "], check=True)"
            )
        return list(additional_imports)

    def _patch_final_answer_with_exception(self, final_answer_tool: Any) -> None:
        # CodeAgent registers its tools on every run. Upstream's patch wraps the
        # existing forward method and must be applied only once to each tool.
        if not getattr(final_answer_tool, "_briq_final_answer_patched", False):
            super()._patch_final_answer_with_exception(final_answer_tool)
            final_answer_tool._briq_final_answer_patched = True

    @staticmethod
    def _deserialize_final_answer(encoded_value: str, allow_pickle: bool = False) -> Any:
        if allow_pickle or not encoded_value.startswith("safe:"):
            raise ValueError("Only JSON final answers are supported")
        pending = [json.loads(encoded_value[5:])]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                # These upstream decoders can expand small wire values into
                # large host allocations. Keep rich objects in the sandbox.
                if value.get("__type__") in ("ndarray", "PIL.Image"):
                    raise ValueError("Return arrays as lists and images as bytes")
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        return RemotePythonExecutor._deserialize_final_answer(encoded_value, allow_pickle=False)

    def cleanup(self) -> None:
        """Destroy owned resources. Failures propagate and can be retried."""
        with self._lock:
            self._poisoned = True
            if self.sandbox is not None:
                try:
                    self.sandbox.destroy()
                except BaseException as error:
                    error.add_note(
                        f"Sandbox cleanup failed; retry destruction for {self.sandbox.id}"
                    )
                    raise
                self.sandbox = None
            if self._owned_client and not self._closed:
                self._client.close()
            self._closed = True

    def delete(self) -> None:
        self.cleanup()

    def __enter__(self) -> BriqExecutor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self.cleanup()
        except BaseException as cleanup_error:
            if exc is not None:
                raise BaseExceptionGroup(
                    "Execution and cleanup failed", [exc, cleanup_error]
                ) from None
            raise
