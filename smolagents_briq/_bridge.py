"""Runs inside the sandbox only; Jupyter messages leave as bounded JSON."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty

from jupyter_client import BlockingKernelClient

LIMIT = 65536


def main():
    root, action = Path(sys.argv[1]), sys.argv[2]
    connection = root / "kernel.json"
    if action == "boot":
        with open(os.devnull, "wb") as null:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    str(connection),
                    "--IPKernelApp.ip=127.0.0.1",
                ],
                stdin=subprocess.DEVNULL,
                stdout=null,
                stderr=null,
                start_new_session=True,
            )
        (root / "pid").write_text(str(process.pid))
        deadline = time.monotonic() + 15
        while not connection.exists():
            if time.monotonic() >= deadline or process.poll() is not None:
                raise RuntimeError("Kernel did not start")
            time.sleep(0.05)
    client = BlockingKernelClient(connection_file=str(connection))
    client.load_connection_file()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=15 if action == "boot" else 5)
        if action == "boot":
            print("{}")
            return
        timeout = float(sys.argv[4])
        message_id = client.execute(Path(sys.argv[3]).read_text())
        deadline = time.monotonic() + timeout
        result = {"logs": "", "output": None, "error": None, "timed_out": False}
        budget = LIMIT
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.kill(int((root / "pid").read_text()), signal.SIGINT)
                result["timed_out"] = True
                break
            try:
                message = client.get_iopub_msg(timeout=min(remaining, 1))
            except Empty:
                continue
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            kind, content = message["msg_type"], message["content"]
            if kind == "status" and content["execution_state"] == "idle":
                break
            if kind == "stream":
                text = content["text"]
                result["logs"] += text[:budget]
                budget -= min(budget, len(text))
            elif kind in ("execute_result", "display_data"):
                text = content["data"].get("text/plain", "")
                result["output"] = text[:budget]
                budget -= min(budget, len(text))
            elif kind == "error":
                if len(content["evalue"]) > LIMIT:
                    result["error"] = {
                        "name": "OutputLimitError",
                        "value": "Error or final answer exceeds 65536 characters",
                    }
                    continue
                result["error"] = {
                    "name": content["ename"][:256],
                    "value": content["evalue"][:LIMIT],
                }
        print(json.dumps(result))
    finally:
        client.stop_channels()


if __name__ == "__main__":
    main()
