"""Regression tests for crash-safe NWB output naming."""

from pathlib import Path
import tempfile
import unittest

from pipeline_gui import nwb_conversion_complete
from rec2nwb.rec2nwb_interp import (
    _prepare_incomplete_nwb,
    _publish_completed_nwb,
    incomplete_nwb_path,
)


class IncompleteNWBGuardTests(unittest.TestCase):
    def test_incomplete_marker_forces_reconversion_even_with_old_final(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "session_sh0.nwb"
            final_path.write_bytes(b"old complete file")

            staging_path = _prepare_incomplete_nwb(final_path)

            self.assertEqual(staging_path, incomplete_nwb_path(final_path))
            self.assertTrue(staging_path.is_file())
            self.assertFalse(nwb_conversion_complete(final_path))

    def test_publish_atomically_replaces_final_and_clears_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "session_sh0.nwb"
            final_path.write_bytes(b"old")
            staging_path = _prepare_incomplete_nwb(final_path)
            staging_path.write_bytes(b"new")

            _publish_completed_nwb(staging_path, final_path)

            self.assertEqual(final_path.read_bytes(), b"new")
            self.assertFalse(staging_path.exists())
            self.assertTrue(nwb_conversion_complete(final_path))

    def test_restart_discards_stale_staging_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "session_sh0.nwb"
            staging_path = incomplete_nwb_path(final_path)
            staging_path.write_bytes(b"partial data")

            prepared_path = _prepare_incomplete_nwb(final_path)

            self.assertEqual(prepared_path.read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
