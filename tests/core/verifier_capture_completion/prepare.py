#!/usr/bin/env python3
"""Source extraction only; compile/run separately under a fresh bounded grant."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
s=a.source.read_text();start=s.index("    const cudaError_t ue = cudaGraphUpload(exec_t, cs_);",s.index("bool Verifier::capture(int T"));end=s.index("\nbool Verifier::capture_commit",start)
body=s[start:end].rstrip();assert body.endswith("}");body=body[:-1]
a.output.write_text(body)
