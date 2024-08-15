import numpy as np


def list_like(value):
    """
    Checks if a given value is list-like.

    Parameters
    ----------
    value : object
        The value to be checked.

    Returns
    -------
    bool
        Returns True if the value is list-like, otherwise returns False.

    Notes
    -----
    A value is considered list-like if it meets the following conditions:
    - It does not have a "strip" attribute.
    - It has either a "__getitem__" attribute or an "__iter__" attribute.
    """
    return (not hasattr(value, "strip")
            and (hasattr(value, "__getitem__")
                 or hasattr(value, "__iter__")))


def norm_l1(x, normalize=False):
    """
    Calculate the L1 norm of an array.

    Parameters
    ----------
    x : numpy.ndarray
        The input array.
    normalize : bool, optional
        Specifies whether to normalize the L1 norm. Default: False.

    Returns
    -------
    float
        The L1 norm of the input array.
    """
    norm = np.sum(np.abs(x))
    if normalize:
        norm /= len(x)
    return norm


def norm_l2(x, normalize=False):
    """
    Parameters
    ----------
    x : numpy.ndarray
        The input array.
    normalize : bool, optional
        Specifies whether to normalize the L2 norm. Default: False.

    Returns
    -------
    float
        The L2 norm of the input array.
    """
    norm = np.sum(np.square(x))
    if normalize:
        norm = np.sqrt(norm / len(x))
    return norm


def norm_lmax(x, normalize=False):
    """
    Calculate the Lmax norm of an input array.

    Parameters
    ----------
    x : numpy.ndarray
        The input array.

    normalize : bool, optional
        Specifies whether to normalize the Lmax norm. Default: False.

    Returns
    -------
    norm : float
        The Lmax norm of the input array.

    Notes
    -----
    The Lmax norm, also known as the infinity norm or maximum norm, is defined
    as the maximum absolute value of the elements in the array.
    """
    norm = np.max(np.abs(x))
    if not normalize:
        norm *= len(x)
    return norm


def norm_lp(x, p, normalize=False):
    """
    Calculates the Lp norm of a given vector.

    Parameters
    ----------
    x : numpy.ndarray
        The input vector for which the Lp norm needs to be calculated.
    p : float
        The Lp norm exponent. It should be a positive number.
    normalize : bool, optional
        Specifies whether to normalize the Lp norm. Default: False.

    Returns
    -------
    float
        The Lp norm of the input vector.
    """
    norm = np.sum(np.abs(x ** p))
    if normalize:
        norm = (norm / len(x)) ** (1.0 / p)
    return norm
