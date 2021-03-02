"""
Python Options
To access Python options
vectorhub.options['catch_vector_error'] = False
"""

OPTIONS = {
    'catch_vector_error': True
}

def get_option(field):
    """Get an option with a specific field name.
    """
    return OPTIONS[field]

def set_option(field, value):
    """Set an option with a specific field name.
    """
    OPTIONS[field] = value
