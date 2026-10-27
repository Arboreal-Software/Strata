"""Shard/identity regression cases without model weights."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from window_compatibility import inspect_model


def string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def shard(path, index, count=2, architecture=None):
    meta = {"split.no": index, "split.count": count}
    if index == 0:
        meta.update({"general.architecture": "qwen4exp", "tokenizer.ggml.tokens": ["a", "b"],
                     "tokenizer.chat_template": "{{ messages }}", "qwen4exp.block_count": 48})
    if architecture:
        meta["general.architecture"] = architecture
    data = struct.pack("<IIQQ", 0x46554747, 3, 0, len(meta))
    for key, value in meta.items():
        data += string(key)
        if isinstance(value, int):
            data += struct.pack("<II", 4, value)
        elif isinstance(value, list):
            data += struct.pack("<IIQ", 9, 8, len(value)) + b"".join(string(s) for s in value)
        else:
            data += struct.pack("<I", 8) + string(value)
    path.write_bytes(data)
    return {"name": path.name, "bytes": len(data), "sha256": "a" * 64}


class CompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = [Path(self.tmp.name) / "first.gguf", Path(self.tmp.name) / "second.gguf"]
        self.receipt = [shard(p, i) for i, p in enumerate(self.paths)]

    def test_secondary_shard_may_omit_metadata_and_order_is_irrelevant(self):
        result = inspect_model(self.paths[::-1], self.receipt)
        self.assertFalse(result["runtime_qualified"])
        self.assertEqual(result["geometry"]["qwen4exp.block_count"], 48)
        self.assertTrue(all(not a["content_hash_verified_this_run"] for a in result["artifacts"]))
        self.assertEqual([a["split_no"] for a in result["artifacts"]], [0, 1])
        json.dumps(result, allow_nan=False)

    def test_missing_and_duplicate_shards(self):
        for paths in [self.paths[:1], [self.paths[0], self.paths[0]]]:
            with self.assertRaises(ValueError):
                inspect_model(paths, self.receipt)

    def test_conflicting_secondary_architecture(self):
        self.receipt[1] = shard(self.paths[1], 1, architecture="other")
        with self.assertRaises(ValueError):
            inspect_model(self.paths, self.receipt)

    def test_receipt_does_not_certify_different_file(self):
        self.receipt[0]["bytes"] += 1
        with self.assertRaises(ValueError):
            inspect_model(self.paths, self.receipt)

    def test_duplicate_number(self):
        self.receipt[1] = shard(self.paths[1], 0)
        with self.assertRaises(ValueError):
            inspect_model(self.paths, self.receipt)


if __name__ == "__main__":
    unittest.main()
