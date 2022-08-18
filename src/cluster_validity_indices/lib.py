"""
Library for the cluster-validity-indices package.
"""

def my_function():
    """
    Prints hello world.
    """
    print("Hello world!")
    return

class ASDF():

    def __init__(self):
        self.data = []

class InvalidKindError(Exception):
    """Raised if the kind is invalid."""
    pass


def get_rest_ingredients(kind:str=None):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]
    """
    return ["shells", "gorgonzola", "parsley"]

def get_napoleon_ingredients(kind:str=None):
    """
    Return a list of random ingredients as strings.

    Parameters
    ----------
    kind : list[str] or None
        Optional "kind" of ingredients.

    Raises
    ------
    lumache.InvalidKindError
        If the kind is invalid.

    Returns
    -------
    list[str]
        The ingredients list.
    """
    return ["shells", "gorgonzola", "parsley"]
