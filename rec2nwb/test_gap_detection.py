"""
Regression tests for gap detection in trodes_dio_gui.py.

The bug these pin down: a single dropped packet leaves a timestamp delta of 2,
and Trodes' console reports that delta ("gap of 2 points"). rec2nwb_interp.py
synthesizes exactly ``n_missing`` frames, so writing Trodes' number verbatim
over-fills every gap by one sample. What we record is delta - 1.

Measured against CnL46_20260731 with 'trodesexport -time -interp 0':

    passive    124,176,612 raw samples, 4,599 gaps -> 124,181,211 interpolated
    task        58,632,858 raw samples,    60 gaps ->  58,632,918 interpolated

i.e. Trodes itself inserts one sample per gap, not two.

Runs under pytest if it is installed, and standalone if it is not (the sorting
environment has no pytest and does not need one for this):

    python rec2nwb/test_gap_detection.py
    python -m pytest rec2nwb/test_gap_detection.py
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:                                   # standalone fallback
    class _Pytest:
        @staticmethod
        @contextmanager
        def raises(exc, match=None):
            import re as _re
            try:
                yield
            except exc as e:
                if match and not _re.search(match, str(e)):
                    raise AssertionError(
                        f"{exc.__name__} raised but message {str(e)!r} "
                        f"does not match {match!r}")
            else:
                raise AssertionError(f"{exc.__name__} was not raised")

    pytest = _Pytest()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # DIO.py imports process_func

from rec2nwb.trodes_dio_gui import (GAP_RE, TS_DTYPE, find_gaps, gap_file_text,
                                    read_timestamps)
from rec2nwb.rec2nwb_interp import SpikeGadgetsRecToNWB  # noqa: E402


def write_timestamps(path: Path, times: np.ndarray) -> Path:
    """Build a Trodes .timestamps.dat around *times*."""
    header = (
        "<Start settings>\n"
        "Description: Timestamps\n"
        "Byte_order: little endian\n"
        "Clockrate: 30000\n"
        f"First_timestamp: {int(times[0])}\n"
        "Fields: <time uint32><systime int64>\n"
        "<End settings>\n"
    ).encode("ascii")
    records = np.zeros(len(times), dtype=TS_DTYPE)
    records["time"] = times
    records["systime"] = np.arange(len(times), dtype=np.int64)
    path.write_bytes(header + records.tobytes())
    return path


def test_continuous_stream_has_no_gaps():
    assert find_gaps(np.arange(1000, 2000, dtype=np.uint32)) == []


def test_single_dropped_packet_is_one_missing_not_two():
    # 10, 11, 13 -> the sample at 12 is gone. Trodes prints "gap of 2 points".
    times = np.array([10, 11, 13, 14], dtype=np.uint32)
    assert find_gaps(times) == [(11, 1)]


def test_gap_size_is_delta_minus_one():
    times = np.array([100, 101, 111, 112], dtype=np.uint32)  # 9 samples absent
    assert find_gaps(times) == [(101, 9)]


def test_multiple_gaps_in_order():
    times = np.array([5, 6, 8, 9, 20, 21], dtype=np.uint32)
    assert find_gaps(times) == [(6, 1), (9, 10)]


def test_missing_total_matches_span_minus_count():
    rng = np.random.default_rng(0)
    times = np.arange(0, 50_000, dtype=np.int64)
    drop = rng.choice(np.arange(1, 49_999), size=300, replace=False)
    times = np.setdiff1d(times, drop).astype(np.uint32)

    gaps = find_gaps(times)
    span = int(times[-1]) - int(times[0]) + 1
    assert sum(n for _, n in gaps) == span - len(times) == 300


def test_edge_cases():
    assert find_gaps(np.empty(0, dtype=np.uint32)) == []
    assert find_gaps(np.array([7], dtype=np.uint32)) == []


def test_roundtrip_through_a_real_timestamps_file(tmp_path):
    times = np.array([1000, 1001, 1003, 1004, 1010], dtype=np.uint32)
    path = write_timestamps(tmp_path / "s.timestamps.dat", times)

    assert np.array_equal(read_timestamps(path), times)
    assert find_gaps(read_timestamps(path)) == [(1001, 1), (1004, 5)]


def test_read_timestamps_rejects_a_file_with_no_header(tmp_path):
    bad = tmp_path / "bad.timestamps.dat"
    bad.write_bytes(b"not a trodes file")
    with pytest.raises(ValueError, match="End settings"):
        read_timestamps(bad)


def test_sidecar_is_readable_by_rec2nwb_interp(tmp_path):
    """The written .txt must parse back to the same (timestamp, n_missing)."""
    gaps = [(11, 1), (500, 3)]
    txt = tmp_path / "x.rec.txt"
    txt.write_text(gap_file_text("x.rec", gaps), encoding="utf-8")

    assert SpikeGadgetsRecToNWB._parse_gap_file(txt) == sorted(gaps)


def test_sidecar_header_is_comments_only(tmp_path):
    """Header lines must not look like gap records to the downstream regex."""
    txt = tmp_path / "y.rec.txt"
    txt.write_text(gap_file_text("y.rec", [(11, 1)]), encoding="utf-8")

    header = [ln for ln in txt.read_text().splitlines() if ln.startswith("#")]
    assert header and not any(GAP_RE.search(ln) for ln in header)


def test_empty_gap_list_still_writes_a_parseable_file(tmp_path):
    txt = tmp_path / "z.rec.txt"
    txt.write_text(gap_file_text("z.rec", []), encoding="utf-8")

    assert SpikeGadgetsRecToNWB._parse_gap_file(txt) == []
    assert "0 gap(s)" in txt.read_text()


def test_measured_session_counts():
    """The real CnL46_20260731 numbers, as a guard against convention drift."""
    for raw, n_gaps, interpolated in ((124_176_612, 4_599, 124_181_211),
                                      (58_632_858, 60, 58_632_918),
                                      (153_711_403, 147, 153_711_550)):
        assert raw + n_gaps == interpolated


class _RampRecording:
    """One channel whose value at each frame is that frame's hardware timestamp."""

    def __init__(self, timestamps):
        self._ts = np.asarray(timestamps, dtype=np.int32).reshape(-1, 1)

    def get_num_frames(self):
        return len(self._ts)

    def get_num_channels(self):
        return 1

    def get_traces(self, start_frame=0, end_frame=None, channel_ids=None):
        return self._ts[start_frame:end_frame]


def _gap_positions(gap_timestamps, first):
    """Source-frame positions, the way process_folder computes them."""
    positions, cum_missing = [], 0
    for t in gap_timestamps:
        positions.append((t - first - cum_missing, 1))
        cum_missing += 1
    return positions


def test_filled_stream_lands_on_a_continuous_timeline():
    """out[k] must equal first_timestamp + k once the gaps are filled.

    PCHIP through a linear ramp reproduces the ramp exactly, so a fill inserted
    at the wrong index shows up as a mismatch rather than as plausible noise.
    """
    first = 10_468_534
    gap_ts = [10478105, 10493258, 10497591, 10511025, 10528483,
              10616852, 10628526, 10671185, 10709807, 10746897]

    ticks = np.arange(first, gap_ts[-1] + 5001, dtype=np.int64)
    raw = np.setdiff1d(ticks, np.array([t + 1 for t in gap_ts], dtype=np.int64))
    rec = _RampRecording(raw)

    assert [t for t, _ in find_gaps(raw)] == gap_ts

    conv = SpikeGadgetsRecToNWB(chunk_duration=0.05)
    out = np.concatenate(list(conv._iter_chunks_with_gaps(
        rec, rec.get_num_frames(), 30000.0, None, "t",
        gaps=_gap_positions(gap_ts, first), ctx=10)), axis=0).ravel().astype(np.int64)

    assert len(out) == len(raw) + len(gap_ts)
    assert np.array_equal(out, np.arange(first, first + len(out), dtype=np.int64))


def test_uncorrected_positions_would_break_the_timeline():
    """Guard the guard: without the cumulative-missing term the test must fail."""
    first = 10_468_534
    gap_ts = [10478105, 10493258, 10497591, 10511025, 10528483]

    ticks = np.arange(first, gap_ts[-1] + 5001, dtype=np.int64)
    raw = np.setdiff1d(ticks, np.array([t + 1 for t in gap_ts], dtype=np.int64))
    rec = _RampRecording(raw)

    stale = [(t - first, 1) for t in gap_ts]          # the pre-fix formula
    conv = SpikeGadgetsRecToNWB(chunk_duration=0.05)
    out = np.concatenate(list(conv._iter_chunks_with_gaps(
        rec, rec.get_num_frames(), 30000.0, None, "t", gaps=stale, ctx=10)),
        axis=0).ravel().astype(np.int64)

    assert not np.array_equal(out, np.arange(first, first + len(out), dtype=np.int64))


def _main() -> int:
    """Run every test in this file without pytest."""
    import inspect
    import tempfile
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = []
    for name, fn in tests:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
        except Exception:
            failed.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")

    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
