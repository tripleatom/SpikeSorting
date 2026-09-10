"""Read NWB recordings and restore saved ProbeInterface contact geometry."""

import json
import numpy as np
from probeinterface import Probe
import spikeinterface.extractors as se


def read_nwb_recording(*args, **kwargs):
    kwargs.setdefault('load_channel_properties', True)
    recording = se.read_nwb_recording(*args, **kwargs)
    required = {'contact_shape', 'contact_shape_params', 'contact_id', 'shank_id'}
    if not required.issubset(recording.get_property_keys()):
        return recording  # Older NWBs contain locations only.
    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(
        recording.get_channel_locations()[:, :2],
        shapes=recording.get_property('contact_shape'),
        shape_params=[json.loads(p) for p in recording.get_property('contact_shape_params')],
        shank_ids=recording.get_property('shank_id'),
    )
    probe.set_contact_ids(recording.get_property('contact_id'))
    probe.set_device_channel_indices(np.arange(recording.get_num_channels()))
    return recording.set_probe(probe, in_place=False)
