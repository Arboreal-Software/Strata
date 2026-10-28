"""Reject stale serial adaptation controller extracts before fixture build."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("driver")
    parser.add_argument("fixtures")
    args = parser.parse_args()
    source = Path(args.driver).read_text()
    fixtures = Path(args.fixtures)
    spans = [
        ("serve.inc", "                std::thread adapt_thr;   // the adaptive tier beside the commit",
         "                if (!ver.commit"),
        ("nonserve.inc", "            std::thread adapt_thr;\n            bool adapt_ok",
         '            strata::core::progress_at("spec loop: commit"'),
    ]
    for name, marker, stop in spans:
        start = source.index(marker)
        end = source.index(stop, start)
        if source[start:end].strip() != (fixtures / name).read_text().strip():
            raise SystemExit("stale serial adaptation extract: " + name)
    nonserve = source.index("            std::thread adapt_thr;\n            bool adapt_ok")
    start = source.index("            if (eos) {", nonserve)
    end = source.index("            const bool drafted", start)
    if source[start:end].strip() != (fixtures / "eos.inc").read_text().strip():
        raise SystemExit("stale serial adaptation extract: eos.inc")


if __name__ == "__main__":
    main()
