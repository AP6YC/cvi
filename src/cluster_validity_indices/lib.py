"""
Library for the cluster-validity-indices package.
"""

def my_function():
    print("Hello world!")
    return

class ASDF():

    def __init__(self):
        self.data = []

class InvalidKindError(Exception):
    """Raised if the kind is invalid."""
    pass


def get_random_ingredients(kind:str=None):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]

    Parameters
    ----------
    kind : list[str]
        stuff and things
    asdf : str
        does something i guess
    """
    return ["shells", "gorgonzola", "parsley"]
