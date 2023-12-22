import importlib.metadata

import hpo2 as m


def test_version():
    assert importlib.metadata.version("hpo2") == m.__version__
