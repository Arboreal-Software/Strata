"""Reject stale driver controller extraction before building this fixture."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("driver")
    parser.add_argument("fixture")
    args = parser.parse_args()
    source = Path(args.driver).read_text()
    start = source.index("        bool pipe_completion_uncertain = false;")
    drain = source.index("        auto pipe_drain = ", start)
    end = source.index("\n#if defined(STRATA_WINDOW_STATE_AVAILABLE)", drain)
    if source[start:end].strip() != Path(args.fixture).read_text().strip():
        raise SystemExit("submitted-work controller fixture is stale; refresh the complete wrapper and drain")


if __name__ == "__main__":
    main()
