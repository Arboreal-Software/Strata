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
        ("serve-controller.inc", "                if (!ver.run(T, window.data(), p, win_pool_fn",
         "                if (!ver.commit"),
        ("nonserve-controller.inc", "            if (!ver.run(T, window.data(), p, &drive_pool_multi",
         '            strata::core::progress_at("spec loop: commit"'),
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
    # Placement/freshness assertion only, not a proof about arbitrary C++ flow.
    for run_marker, thread_marker, stop in [
        ("                if (!ver.run(T, window.data(), p, win_pool_fn", "                std::thread adapt_thr;   //", "                if (!ver.commit"),
        ("            if (!ver.run(T, window.data(), p, &drive_pool_multi", "            std::thread adapt_thr;\n            bool adapt_ok", '            strata::core::progress_at("spec loop: commit"'),
    ]:
        if source.count(run_marker) != 1 or source.count(thread_marker) != 1:
            raise SystemExit("ambiguous serial integration boundary")
        begin = source.index(run_marker)
        if not begin < source.index(thread_marker) < source.index(stop, begin):
            raise SystemExit("serial worker moved outside preceding verifier integration")
    nonserve = source.index("            std::thread adapt_thr;\n            bool adapt_ok")
    start = source.index("            if (eos) {", nonserve)
    end = source.index("            const bool drafted", start)
    if source[start:end].strip() != (fixtures / "eos.inc").read_text().strip():
        raise SystemExit("stale serial adaptation extract: eos.inc")


if __name__ == "__main__":
    main()
