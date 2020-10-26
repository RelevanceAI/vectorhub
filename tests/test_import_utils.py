"""
    Test the import utilities.
"""
from vectorhub.import_utils import *
import unittest

def assert_lists_contain_same_elements(list_1, list_2):
    case = unittest.TestCase()
    assert case.assertCountEqual(list_1, list_2) is None

def test_get_requirements():
    requirements = get_requirements()
    assert 'numpy' in requirements.keys()

def test_get_package_requirements():
    assert_lists_contain_same_elements(get_package_requirements('encoders-text-tf-transformers'), ['tensorflow', 'transformers'])
