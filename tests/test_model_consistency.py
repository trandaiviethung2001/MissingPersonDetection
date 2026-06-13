import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import config
from utils.face_recognizer import FaceRecognizer


class FakeFace:
    def __init__(self, embedding):
        self.embedding = embedding
        self.bbox = np.array([0.0, 0.0, 10.0, 10.0])
        self.det_score = 0.99


class FakeFaceAnalysis:
    def __init__(self, name, providers):
        self.name = name
        self.providers = providers
        self.calls = []

    def prepare(self, ctx_id, det_size):
        self.calls.append((ctx_id, det_size))

    def get(self, image):
        return [FakeFace(np.ones(512, dtype=np.float32))]


class ModelConsistencyTests(unittest.TestCase):
    def test_build_database_uses_runtime_insightface_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            person_dir = Path(tmpdir) / 'person_001'
            person_dir.mkdir()
            (person_dir / 'name.txt').write_text('Morgan Freeman', encoding='utf-8')
            (person_dir / 'photo.jpg').write_bytes(b'fake-image')

            output_path = Path(tmpdir) / 'embeddings.pkl'

            with patch('utils.face_recognizer.FaceAnalysis', side_effect=lambda name, providers: FakeFaceAnalysis(name, providers)) as face_analysis, \
                 patch('utils.face_recognizer.cv2.imread', return_value=np.zeros((16, 16, 3), dtype=np.uint8)):
                FaceRecognizer.build_database(str(tmpdir), str(output_path), app=None)

            self.assertEqual(face_analysis.call_args.kwargs['name'], config.INSIGHTFACE_MODEL)
            self.assertIn('providers', face_analysis.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
