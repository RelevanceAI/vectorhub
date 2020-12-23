from vectorhub.auto_encoder import *

def test_get_model_definitions():
    assert isinstance(get_model_definitions(json_fn=None), list)
    assert isinstance(get_model_definitions(json_fn=None)[0], dict)
    assert len(get_model_definitions(json_fn=None)) > 0
