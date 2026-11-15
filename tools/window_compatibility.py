"""Record a candidate window-state profile without loading weights or using a GPU.

Header hashes identify metadata only. Artifact receipt hashes remain reported
evidence until the actual runtime verifies the immutable model manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gguf_reader import GGUFFile


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inspect_model(paths: list[Path], receipt: list[dict]) -> dict:
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("supply distinct model shards")
    receipts = {r["name"]: r for r in receipt}
    if len(receipts) != len(receipt):
        raise ValueError("duplicate artifact receipt")
    artifacts, metadata, tensors = [], [], []
    shard_ids = set()
    for path in paths:
        before = path.stat()
        gguf = GGUFFile(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("artifact changed during header inspection")
        if gguf.metadata.get("general.architecture", "qwen4exp") != "qwen4exp":
            raise ValueError("initial window-state profile requires qwen4exp")
        count = gguf.metadata.get("split.count", 1)
        number = gguf.metadata.get("split.no", 0)
        if count != len(paths) or number in shard_ids or not 0 <= number < count:
            raise ValueError("incomplete or duplicate shard set")
        shard_ids.add(number)
        reported = receipts.get(path.name)
        if reported is None or reported.get("bytes") != after.st_size:
            raise ValueError("artifact receipt missing or byte length differs")
        sha = reported.get("sha256", "")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("artifact receipt lacks SHA-256")
        artifacts.append({"path": str(path.resolve()), "bytes": after.st_size,
                          "mtime_ns": after.st_mtime_ns, "reported_sha256": sha,
                          "content_hash_verified_this_run": False,
                          "header_metadata_sha256": digest(gguf.metadata), "split_no": number})
        metadata.append(gguf.metadata)
        tensors.extend({"name": t.name, "shape": t.shape, "type": t.type_name} for t in gguf.tensors)
    if len({t["name"] for t in tensors}) != len(tensors):
        raise ValueError("duplicate tensor names across shards")
    primary = metadata[next(i for i, a in enumerate(artifacts) if a["split_no"] == 0)]
    if primary.get("general.architecture") != "qwen4exp":
        raise ValueError("primary shard must identify qwen4exp")
    geometry = {k: v for k, v in primary.items() if k.startswith("qwen4exp.")}
    for m in metadata:
        for key, value in geometry.items():
            if key in m and m[key] != value:
                raise ValueError(f"inconsistent shard geometry: {key}")
    tokenizer = {k: v for k, v in primary.items() if k.startswith("tokenizer.") and "chat_template" not in k}
    templates = {k: v for k, v in primary.items() if k.startswith("tokenizer.") and "chat_template" in k}
    if not tokenizer.get("tokenizer.ggml.tokens") or not templates:
        raise ValueError("missing tokenizer or embedded template identity")
    return {
        "schema": "strata-window-candidate-profile-v1", "runtime_qualified": False,
        "artifacts": sorted(artifacts, key=lambda a: a["split_no"]),
        "architecture": "qwen4exp", "geometry": geometry,
        "tokenizer_metadata_sha256": digest(tokenizer),
        "embedded_template_metadata_sha256": digest(templates),
        "tensor_inventory_sha256": digest(sorted(tensors, key=lambda t: t["name"])),
        "tensor_count": len(tensors),
        "unresolved": ["verify actual loaded artifact content identity", "resolved rendered template and overrides",
                       "resolved runtime numeric and speculation settings", "per-device topology and native state layout",
                       "resource admission limits", "single/dual/concurrent runtime qualification"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    result = inspect_model(args.shards, json.loads(args.receipt.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
