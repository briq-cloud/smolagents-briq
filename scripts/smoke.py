"""One-sandbox live contract check. Requires explicit briq API configuration."""

import os

from briq_cloud import Briq, BriqError
from smolagents.default_tools import FinalAnswerTool

from smolagents_briq import BriqExecutor


def main() -> None:
    # Require an explicit endpoint rather than falling back to production.
    base_url = os.environ["BRIQ_API_URL"]
    with Briq(base_url=base_url) as client:
        with BriqExecutor(client=client, execution_timeout=1) as executor:
            sandbox_id = executor.sandbox.id
            executor.send_variables({"number": 21})
            executor("answer = number * 2")
            if executor("answer").output != "42":
                raise RuntimeError("Persistent state check failed")
            executor.send_tools({"final_answer": FinalAnswerTool()})
            result = executor("final_answer(answer)")
            if not result.is_final_answer or result.output != 42:
                raise RuntimeError("Final answer check failed")
            try:
                executor("import time; time.sleep(10)")
            except TimeoutError:
                pass
            else:
                raise RuntimeError("Timeout check failed")
            try:
                executor("1")
            except RuntimeError:
                pass
            else:
                raise RuntimeError("Uncertain executor was reused")
        try:
            deleted = client.sandbox(sandbox_id)
        except BriqError as error:
            if error.code != "not_found":
                raise
        else:
            if deleted.info.state != "destroyed":
                raise RuntimeError("Sandbox is not destroyed after cleanup")
    print("Live checks passed: persistent state, final answer, timeout, reuse refusal, deletion")


if __name__ == "__main__":
    main()
