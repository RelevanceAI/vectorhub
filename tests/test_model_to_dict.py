from vectorhub.auto_encoder import *

def test_get_model_definitions():
    assert isinstance(get_model_definitions(), list)
    assert isinstance(get_model_definitions()[0], dict)
    assert len(get_model_definitions()) > 0
