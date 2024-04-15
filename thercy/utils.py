import numpy as np


def list_like(value):
    """
    Check if a variable is list-like.

    Parameters
    ----------
    value : any
        Variable to check.

    Returns
    -------
    bool
        `True` if the variable is list-like, `False` otherwise.

    """
    return (not hasattr(value, "strip")
            and (hasattr(value, "__getitem__")
                 or hasattr(value, "__iter__")))


def norm_l1(x: np.ndarray, rescale=False):
    norm = np.sum(np.abs(x))

    if rescale:
        norm /= len(x)

    return norm


def norm_l2(x: np.ndarray, rescale=False):
    norm = np.sum(np.square(x))

    if rescale:
        norm = np.sqrt(norm / len(x))

    return norm


def norm_lmax(x: np.ndarray, rescale=False):
    norm = np.max(np.abs(x))

    if not rescale:
        norm *= len(x)

    return norm


def norm_lp(x: np.ndarray, p: float, rescale=False):
    norm = np.sum(np.abs(x ** p))

    if rescale:
        norm = (norm / len(x)) ** (1.0 / p)

    return norm
