"""Regression checks for acquisition wiring and ProbeInterface persistence."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from probeinterface import Probe, write_probeinterface
from pynwb import NWBHDF5IO
from spikeinterface import NumpyRecording

from rec2nwb.probes import load_probes, mapping_dir, list_device_types, probe_for_channels
from rec2nwb.utils.electrode import get_all_shanks, get_ch_index_on_shank, build_electrode_df
from rec2nwb.utils.nwb_helpers import make_nwbfile, add_electrodes_to_nwb, make_electrical_series
from rec2nwb.nwb_recording import read_nwb_recording


class ProbeMappingTests(unittest.TestCase):
    def test_converted_probeinterface_maps(self):
        cases = [('4shank16intan', 15, 128, 4),
                 ('4shank32', 20, 128, 4),
                 ('4shank32intan', 20, 128, 4),
                 ('4shank32rp', 20, 128, 4),
                 ('8shank32', 20, 256, 8)]
        for name, width, contact_count, shank_count in cases:
            with self.subTest(name=name):
                probes = load_probes(name)
                self.assertEqual(len(probes), 1)
                source = probes[0]
                self.assertEqual(source.get_contact_count(), contact_count)
                self.assertEqual(get_all_shanks(name), list(range(shank_count)))
                position_by_channel = {
                    int(channel): position
                    for channel, position in zip(source.device_channel_indices,
                                                 source.contact_positions)
                    if channel >= 0
                }
                for shank in get_all_shanks(name):
                    ch, x, y = get_ch_index_on_shank(shank, name)
                    expected_positions = np.asarray([position_by_channel[int(i)] for i in ch])
                    np.testing.assert_array_equal(x, expected_positions[:, 0])
                    np.testing.assert_array_equal(y, expected_positions[:, 1])
                    # Mimic dropped bad channels and a reordered recording.
                    selected = ch[::2][::-1]
                    probe = probe_for_channels(name, selected)
                    np.testing.assert_array_equal(probe.contact_positions,
                                                  [position_by_channel[int(i)] for i in selected])
                    self.assertTrue(all(s == 'square' for s in probe.contact_shapes))
                    self.assertTrue(all(p['width'] == width for p in probe.contact_shape_params))
                    recording = NumpyRecording(np.zeros((10, len(selected))), 30000)
                    attached = recording.set_probe(probe)
                    np.testing.assert_array_equal(attached.get_channel_locations(), probe.contact_positions)

    def test_explicit_wiring_json_only_and_disconnected_contacts(self):
        with tempfile.TemporaryDirectory() as directory:
            probe = Probe(ndim=2, si_units='um')
            probe.set_contacts([[0, 0], [0, 25], [0, 50]], shapes='square',
                               shape_params={'width': 15}, shank_ids=['2', '2', '7'])
            probe.set_device_channel_indices([9, 3, -1])
            probe.set_contact_ids(['a', 'b', 'c'])
            write_probeinterface(Path(directory) / 'custom.probeinterface.json', probe)
            with patch('rec2nwb.probes.mapping_dir', return_value=Path(directory)):
                self.assertEqual(list_device_types(), ['custom'])
                self.assertEqual(get_all_shanks('custom'), [2])
                ch, x, y = get_ch_index_on_shank(2, 'custom')
                np.testing.assert_array_equal(ch, [3, 9])
                np.testing.assert_array_equal(y, [25, 0])
                sliced = probe_for_channels('custom', [9, 3])
                np.testing.assert_array_equal(sliced.contact_ids, ['a', 'b'])
                np.testing.assert_array_equal(sliced.device_channel_indices, [0, 1])

    def test_nwb_roundtrip(self):
        name = '4shank16intan'
        ch, x, y = get_ch_index_on_shank(0, name)
        electrodes = build_electrode_df(ch, x, y, 'intan', bad_ch_ids=[f'ch{ch[1]}'], device_type=name)
        nwb = make_nwbfile(datetime.now(timezone.utc), {})
        region = add_electrodes_to_nwb(nwb, electrodes, 0, 'test')
        traces = np.arange(20 * len(electrodes), dtype=np.int16).reshape(20, -1)
        nwb.add_acquisition(make_electrical_series(traces, region, 30000, 1e-6, 0.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.nwb'
            with NWBHDF5IO(path, 'w') as io:
                io.write(nwb)
            recording = read_nwb_recording(path)
            np.testing.assert_array_equal(recording.get_traces(), traces)
            probe = recording.get_probe()
            np.testing.assert_array_equal(probe.contact_positions, electrodes[['x', 'y']])
            self.assertTrue(all(s == 'square' for s in probe.contact_shapes))
            self.assertTrue(all(p['width'] == 15 for p in probe.contact_shape_params))
            # Close the underlying extractor before Windows removes the fixture.
            recording._file.close()
            del recording._file

    def test_legacy_csv_geometry(self):
        table = pd.read_csv(mapping_dir() / '4shank16.csv')
        for shank in get_all_shanks('4shank16'):
            ch, x, y = get_ch_index_on_shank(shank, '4shank16')
            np.testing.assert_array_equal(ch, table[table.sh == shank].index)
            np.testing.assert_array_equal(x, table.loc[ch, 'xcoord'])
            np.testing.assert_array_equal(y, table.loc[ch, 'ycoord'])

    def test_plain_json_and_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.json'
            probe = Probe(ndim=2, si_units='um')
            probe.set_contacts([[0, 0]], shapes='square', shape_params={'width': 20}, shank_ids=['0'])
            probe.set_device_channel_indices([7])
            write_probeinterface(path, probe)
            (Path(directory) / 'custom.csv').write_text('invalid CSV must not be loaded')
            (Path(directory) / 'kilosort.json').write_text('{"chanMap": [0]}')
            with patch('rec2nwb.probes.mapping_dir', return_value=Path(directory)):
                self.assertEqual(list_device_types(), ['custom'])
                for name in ('custom', 'custom.json', str(path)):
                    np.testing.assert_array_equal(load_probes(name)[0].device_channel_indices, [7])


if __name__ == '__main__':
    unittest.main()
