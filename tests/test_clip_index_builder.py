"""Resuming an index build must preserve the exact embedding bytes."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from fashion_how_graphdb import clip_onnx as clip, web_service as service
from scripts import build_clip_index as builder


class IndexBuilderTests(unittest.TestCase):
    def test_resuming_keeps_identical_vectors_without_reencoding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "1_0.jpg"
            image.write_bytes(b"test-image")
            vector = clip.normalized_vectors(np.arange(1, 513, dtype=np.float32).reshape(1, -1))
            with patch.object(builder, "ROOT", root), \
                 patch.object(service, "image_index", return_value={image.name: image}), \
                 patch.object(service, "sample_ids", return_value={"dresses": ["1_0"]}), \
                 patch.object(clip, "encode_images", return_value=vector) as encode, \
                 patch.object(sys, "argv", ["build", "--output-dir", str(root / "index")]), \
                 redirect_stdout(StringIO()):
                builder.main()
                first_vectors = (root / "index/image_embeddings.npy").read_bytes()
                first_manifest = (root / "index/manifest.json").read_bytes()
                builder.main()
                self.assertEqual(first_vectors, (root / "index/image_embeddings.npy").read_bytes())
                self.assertEqual(first_manifest, (root / "index/manifest.json").read_bytes())
                encode.assert_called_once()


if __name__ == "__main__":
    unittest.main()
