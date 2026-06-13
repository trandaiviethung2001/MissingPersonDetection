import importlib
import unittest


class DetectorServiceImportTests(unittest.TestCase):
    def test_detector_service_helpers_are_available(self):
        detect_missing_person = importlib.import_module('detect_missing_person')

        self.assertTrue(hasattr(detect_missing_person, 'open_video_capture'))
        self.assertTrue(hasattr(detect_missing_person, 'capture_fourcc'))
        self.assertTrue(callable(detect_missing_person.open_video_capture))
        self.assertTrue(callable(detect_missing_person.capture_fourcc))


if __name__ == '__main__':
    unittest.main()
