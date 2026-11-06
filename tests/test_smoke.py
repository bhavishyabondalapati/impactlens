import importlib

import pytest

import impactlens


def test_version():
    assert impactlens.__version__ == "0.1.0"


@pytest.mark.parametrize("sub", ["data", "models", "geo", "agents", "aws"])
def test_subpackages_import(sub):
    assert importlib.import_module(f"impactlens.{sub}")
