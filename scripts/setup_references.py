"""Fetch the pinned xVerify sources whose judge prompt and template label every response."""
from __future__ import annotations

import subprocess
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "third_party" / "xverify_official"
URL = "https://github.com/IAAR-Shanghai/xVerify.git"
COMMIT = "3f50a37912992ab89c51bdc8a83a655c85714079"


def git(*args):
    return subprocess.check_output(["git", "-C", str(TARGET), *args], text=True).strip()


if __name__ == "__main__":
    if not TARGET.exists():
        TARGET.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(TARGET)], check=True)
        subprocess.run(["git", "-C", str(TARGET), "remote", "add", "origin", URL], check=True)
        subprocess.run(["git", "-C", str(TARGET), "fetch", "-q", "--depth", "1", "origin", COMMIT], check=True)
        subprocess.run(["git", "-C", str(TARGET), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    if git("rev-parse", "HEAD") != COMMIT:
        raise SystemExit(f"{TARGET} is not at {COMMIT}; remove it and rerun")
    print(f"xVerify {COMMIT[:7]} ready at {TARGET}")
