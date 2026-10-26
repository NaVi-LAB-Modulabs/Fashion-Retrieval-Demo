"""Run offline tests and publish concise CI failure annotations."""

from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if os.getenv("GITHUB_ACTIONS") == "true":
        for test, traceback in [*result.failures, *result.errors]:
            # Test identifiers and assertion summaries are enough to diagnose a
            # failure from check annotations without access to full runner logs.
            summary = f"{test.id()}: {traceback.strip().splitlines()[-1]}"
            escaped = summary.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::error::{escaped}", flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
