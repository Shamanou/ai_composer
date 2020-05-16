import unittest

from src.lstm import list_blobs_with_prefix, get_instruments


class TestLstm(unittest.TestCase):
    def test_list_blobs_with_prefix(self):
        blob_list = list_blobs_with_prefix("midi_shamanou", "new_midi_songs")

        self.assertGreater(len(blob_list), 0)