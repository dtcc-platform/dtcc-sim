"""Synthesis of reproducible regional simulation days."""

from hashlib import blake2s, sha256
from pathlib import Path

import numpy as np


def rng(seed, *keys):
    """Independent deterministic stream for one scenario entity and stage."""
    key = ":".join(map(str, (seed, *keys))).encode()
    return np.random.default_rng(
        int.from_bytes(blake2s(key, digest_size=16).digest(), "little")
    )


def fingerprint(path):
    """Return the immutable SHA-256 of one archived upstream artifact."""
    with Path(path).open("rb") as stream:
        from hashlib import file_digest

        return file_digest(stream, sha256).hexdigest()
