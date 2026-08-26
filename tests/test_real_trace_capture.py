from benchmarks.capture_real_trace import (
    capture_bsearch,
    capture_matmul,
    to_champsim_format,
)
from nncpu.traces import read_champsim


class BufferRecorder:
    def __init__(self):
        self.instructions = []

    def load(self, address):
        self.instructions.append({"type": "LOAD", "address": address})

    def store(self, address, value):
        self.instructions.append({"type": "STORE", "address": address, "value": value})


def test_binary_search_capture_uses_multiple_targets():
    recorder = BufferRecorder()
    capture_bsearch(recorder, n=256)
    touched = {inst["address"] for inst in recorder.instructions}
    # Reinitializing Random(11) for every target used to repeat one identical
    # search path and touched fewer than ten addresses.
    assert len(touched) > 100


def test_matmul_records_read_modify_write_of_output():
    recorder = BufferRecorder()
    capture_matmul(recorder, n=2)
    types = [inst["type"] for inst in recorder.instructions]
    assert types.count("LOAD") == 2 * 2 + 2 * 2 * 2 * 2
    assert types.count("STORE") == 2 * 2 * 2


def test_champsim_export_round_trips_word_addresses(tmp_path):
    original = [
        {"type": "LOAD", "address": 0x2000},
        {"type": "STORE", "address": 0x2001, "value": 7},
    ]
    path = tmp_path / "roundtrip.trace"
    path.write_text(to_champsim_format(original), encoding="utf-8")
    parsed = read_champsim(str(path))
    assert [i["address"] for i in parsed] == [i["address"] for i in original]
    assert parsed[1]["value"] == 0  # access traces intentionally omit data values
