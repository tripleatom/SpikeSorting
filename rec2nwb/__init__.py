"""rec2nwb — recording conversion and direct-sort recording extractors.

``__version__`` is required because this package defines SpikeInterface
extractors (e.g. ``direct_recording.LazyGapInterpolatedRecording``). When SI
serializes a recording graph for ``clone()`` / ``save()`` it records the
defining module's version via ``getattr(module, "__version__", "unknown")``,
and on reload ``_check_same_version`` calls ``packaging.version.parse`` on it.
Without this attribute that parse raises ``AttributeError`` and any
``set_probe``/``clone`` of a rec2nwb recording fails.
"""

__version__ = "0.1.0"
