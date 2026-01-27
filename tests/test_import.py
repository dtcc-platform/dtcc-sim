from dtcc_sim import __version__


def test_version_available():
    assert isinstance(__version__, str)
