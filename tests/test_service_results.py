import tarfile
from pathlib import Path

import pytest

from service import results


class _SavableResult:
    def __init__(self, writer):
        self._writer = writer

    def save(self, target):
        self._writer(Path(target))


@pytest.fixture(autouse=True)
def _shared_results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "SHARED_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(results, "_fsync_file_and_dir", lambda path: None)
    return tmp_path


def test_handle_result_writes_bytes_result(tmp_path):
    meta = results.handle_result(b"mesh-bytes", "pb", "job-1")

    assert meta == {"result_file": "job-1.pb", "size_bytes": 10}
    assert (tmp_path / "job-1.pb").read_bytes() == b"mesh-bytes"


def test_handle_result_moves_single_output_file(tmp_path):
    def write_single_file(target):
        target.write_bytes(b"pb-data")

    meta = results.handle_result(_SavableResult(write_single_file), "pb", "job-2")

    assert meta == {"result_file": "job-2.pb", "size_bytes": 7}
    assert (tmp_path / "job-2.pb").read_bytes() == b"pb-data"


def test_handle_result_archives_multi_file_output(tmp_path):
    def write_xdmf_bundle(target):
        target.write_text("<xdmf />")
        target.with_suffix(".h5").write_bytes(b"hdf5-bytes")

    meta = results.handle_result(_SavableResult(write_xdmf_bundle), "xdmf", "job-3")

    assert meta["result_file"] == "job-3.tar.gz"
    archive = tmp_path / "job-3.tar.gz"
    assert archive.exists()

    with tarfile.open(archive, "r:gz") as tar:
        assert sorted(tar.getnames()) == ["data.h5", "data.xdmf"]


def test_handle_result_requires_format_for_non_bytes():
    with pytest.raises(ValueError, match="format_ext is required"):
        results.handle_result(_SavableResult(lambda target: target.write_bytes(b"x")), None, "job-4")


def test_handle_result_rejects_object_without_save():
    with pytest.raises(TypeError, match="has no save"):
        results.handle_result(object(), "pb", "job-5")


def test_handle_result_rejects_empty_serialization_output():
    with pytest.raises(RuntimeError, match="produced no files"):
        results.handle_result(_SavableResult(lambda target: None), "pb", "job-6")
