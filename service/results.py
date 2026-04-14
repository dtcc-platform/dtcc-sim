import os
import shutil
import tarfile
import tempfile
import logging
from pathlib import Path

from service.config import SHARED_RESULTS_DIR

logger = logging.getLogger(__name__)


def _fsync_file_and_dir(path):
    """Fsync both the file and its parent directory for full durability."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    dirfd = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)


def _serialize_with_save(result, format_ext, tmpdir):
    """Serialize a dataset result into a temporary directory."""
    if not hasattr(result, "save"):
        raise TypeError(
            f"Result object of type {type(result).__name__} has no save() method"
        )

    # Pass str, not Path -- dolfinx monkeypatched save() methods
    # call filename.endswith() which is a str method.
    tmpfile = str(Path(tmpdir) / f"data.{format_ext}")
    result.save(tmpfile)

    files = sorted(Path(tmpdir).iterdir())
    if not files:
        raise RuntimeError(
            f"Result serialization for format '{format_ext}' produced no files"
        )
    return files


def handle_result(result, format_ext, task_id):
    """Write result to shared volume. Archive multi-file outputs as .tar.gz."""
    os.makedirs(SHARED_RESULTS_DIR, exist_ok=True)

    if isinstance(result, bytes):
        # Already serialized (dataset returned bytes when format was set)
        filename = f"{task_id}.{format_ext}" if format_ext else task_id
        path = os.path.join(SHARED_RESULTS_DIR, filename)
        with open(path, "wb") as f:
            f.write(result)
            f.flush()
            os.fsync(f.fileno())
        _fsync_file_and_dir(path)
        return {"result_file": filename, "size_bytes": len(result)}

    # Result is a Python object -- write via save() then check for multi-file output
    if not format_ext:
        raise ValueError(
            f"Cannot serialize result for task {task_id}: "
            f"format_ext is required for non-bytes results"
        )
    tmpdir = tempfile.mkdtemp()
    try:
        files = _serialize_with_save(result, format_ext, tmpdir)

        if len(files) == 1:
            filename = f"{task_id}.{format_ext}"
            dest = os.path.join(SHARED_RESULTS_DIR, filename)
            shutil.move(str(files[0]), dest)
        else:
            filename = f"{task_id}.tar.gz"
            dest = os.path.join(SHARED_RESULTS_DIR, filename)
            with tarfile.open(dest, "w:gz") as tar:
                for f in files:
                    tar.add(str(f), arcname=f.name)

        # Ensure durability before reporting completion
        dest = os.path.join(SHARED_RESULTS_DIR, filename)
        _fsync_file_and_dir(dest)

        size = os.path.getsize(dest)
        return {"result_file": filename, "size_bytes": size}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
