"""Fetch and verify the exact upstream revisions used by reference tests."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
REFERENCES = (
    ("reasc_official", "https://github.com/junseokkim00/ReASC.git",
     "8ee1c76ea0c04bb3cdb81d09bc2655dbc3a8cecb"),
    ("xverify_official", "https://github.com/IAAR-Shanghai/xVerify.git",
     "3f50a37912992ab89c51bdc8a83a655c85714079"),
)


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def setup(*, check: bool = False) -> None:
    for name, url, commit in REFERENCES:
        target = PROJECT / "third_party" / name
        if not target.exists():
            if check:
                raise SystemExit(f"Missing {target}; run scripts/setup_references.py")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.mkdir()
            subprocess.run(["git", "init", str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "remote", "add", "origin", url], check=True)
            subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", commit], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)
        if not (target / ".git").exists():
            raise SystemExit(f"Expected an upstream Git checkout at {target}")
        actual = git(target, "rev-parse", "HEAD")
        if actual != commit:
            raise SystemExit(f"{name}: expected {commit}, found {actual}; existing checkout left untouched")
        if git(target, "status", "--porcelain", "--untracked-files=no"):
            raise SystemExit(f"{name}: tracked upstream files have local changes")
        print(f"{name}: {commit} verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify local checkouts without network access")
    setup(check=parser.parse_args().check)
