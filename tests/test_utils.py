from vectorhub.utils import *

def test_list_models():
    assert len(list_models()) > 0

def test_list_installed_models():
    # Vector AI deployed models should be immediately usable
    assert len(list_installed_models()) > 0
