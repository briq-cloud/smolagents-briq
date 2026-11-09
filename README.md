# smolagents-briq

```python
from smolagents import CodeAgent
from smolagents_briq import BriqExecutor

# Set BRIQ_API_KEY. Supply your configured smolagents Model instance.
with BriqExecutor() as executor:
    agent = CodeAgent(tools=[], model=model, executor=executor)
    result = agent.run("Calculate the first 20 Fibonacci numbers.")
```

Install with `pip install smolagents-briq`. This source candidate targets
smolagents 1.26.x and briq-cloud 0.1.x (minimum 0.1.1). Direct `executor=` injection
works in released smolagents; no plugin discovery or upstream patch is required.
The model is configured separately; no model credentials are sent by this adapter.

Each executor creates and owns a briq sandbox. Python state survives across
agent actions. Tools and variables use smolagents' remote-executor protocol;
tools execute inside the sandbox and must have serializable source and dependencies.
No host tool callback is provided. Expression outputs use Jupyter's text representation.
Final answers use smolagents' safe serialization, with pickle always disabled.
Return arrays as lists and images as bytes. Rich array and image reconstruction
on the host is rejected, including nested values.

The default image is `python:3.12-slim`. Cold setup installs smolagents 1.26.0,
NumPy 1.26/2.x, ipykernel 6/7 and jupyter-client 8 with pip, requiring outbound package access.
A custom image must provide `python3` and pip. Initial installation has a 60-second
command deadline and can fail on slow networks; setup failures destroy the sandbox.
There is no immutable prebuilt integration image in this candidate.

The kernel listens on loopback inside the sandbox. No port is exposed. Commands
and results travel through the briq SDK. Code is uploaded as a file, so long actions
do not become long shell commands. Logs and text results share a 65,536-character
budget per action; rich display formats are not returned. Final answers are limited
to 65,536 characters of serialized data. No remote text is evaluated on the host.

`execution_timeout` defaults to 30 seconds (maximum 40). Timeout interrupts the
kernel and marks the executor unusable. A transport failure or malformed response
also makes it unusable: create a new executor rather than replaying uncertain code.
The adapter serializes execution calls; use one executor per agent.

Use the context manager or call `cleanup()`/`delete()` explicitly. Cleanup destroys
the sandbox, propagates failures and can be retried. An injected `client=Briq(...)`
stays open. There is no destructor that performs network calls. `size` is passed
through when supplied; otherwise the API chooses its default. `sleep` defaults to
`30m`. Sleeping or restarting a sandbox loses the live kernel session.

## Verify this candidate

```sh
uv sync --group test
uv run pytest
uv run ruff check .
uv run python -m build
uv run twine check dist/*
```

Offline contract tests run a real Jupyter kernel and the shipped bridge through a
local subprocess SDK double, plus a deterministic real CodeAgent. They do not call
briq or a model API. A live briq lifecycle smoke is still required before release,
including verifying detached kernel survival after SDK exec, cold setup deadlines,
timeout interruption and sandbox destruction. This candidate has not been live tested.
