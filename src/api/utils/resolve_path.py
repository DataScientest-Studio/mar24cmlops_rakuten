import os

def resolve_path(relative_path):
    """
    Resolves a relative path to an absolute path based on the project's base directory.

    Args:
        relative_path (str): The relative path to resolve.

    Returns:
        str: The absolute path.
    """
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    return os.path.abspath(os.path.join(base_path, relative_path))