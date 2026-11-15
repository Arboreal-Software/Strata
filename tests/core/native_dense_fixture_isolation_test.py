"""Overlap two actual CPU loader fixtures and verify independent retirement."""
import argparse
import contextlib
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def ready_path(child):
    deadline = time.monotonic() + 10
    line = bytearray()
    while b"\n" not in line:
        left = deadline - time.monotonic()
        require(left > 0 and select.select([child.stdout], [], [], max(0, left))[0],
                "fixture readiness timeout")
        chunk = os.read(child.stdout.fileno(), 1)
        require(chunk, "fixture exited before publishing readiness")
        line.extend(chunk)
        require(len(line) <= 4096, "oversized fixture readiness")
    path = Path(line.decode().strip())
    require(path.is_file(), "child did not publish an existing fixture")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("--throw", action="store_true")
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve())
    children, paths = [], []
    with contextlib.ExitStack() as stack:
        errors = []
        try:
            for _ in range(2):
                error = stack.enter_context(tempfile.TemporaryFile())
                errors.append(error)
                children.append(subprocess.Popen(
                    [binary, "--fixture-throw" if args.throw else "--fixture-ready"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error, bufsize=0))
            paths = [ready_path(child) for child in children]
            require(paths[0] != paths[1], "concurrent fixtures share pathname")
            require(not os.path.samefile(*paths), "concurrent fixtures share inode")
            expected = paths[1].read_bytes()
            require(paths[0].read_bytes() == expected, "synthetic inputs differ")
            for i, child in enumerate(children):
                output, _ = child.communicate(b"g", timeout=30)
                errors[i].seek(0)
                error = errors[i].read().decode(errors="replace")
                require(child.returncode == (1 if args.throw else 0),
                        output.decode(errors="replace") + error)
                if args.throw:
                    require("injected fixture unwind" in error, "expected unwind was not exercised")
                require(not paths[i].exists(), "retired fixture was not removed")
                if i == 0:
                    require(paths[1].read_bytes() == expected, "first child damaged second fixture")
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=5)
            # Do not unlink a path merely because a broken binary printed it.
            # Forced termination may leave that child's acquired file for diagnosis.


if __name__ == "__main__":
    main()
