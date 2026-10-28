"""Reject a stale PLE implementation seam before building the combined fixture."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("fixture")
    args = parser.parse_args()
    source = Path(args.source).read_text()
    start = source.index("struct PleReadContext::Impl {")
    end = source.index("\n}  // namespace strata::kernels", start)
    fixture = Path(args.fixture).read_text()
    fixture_start = fixture.index("struct PleReadContext::Impl {")
    if source[start:end].strip() != fixture[fixture_start:].strip():
        raise SystemExit("PLE fixture extraction is stale; copy the complete current implementation before testing")


if __name__ == "__main__":
    main()
