""" Common utilities.
Written by: Jeremie X. J. Bannwarth
"""

import numpy as np


def tilde(v):
    """Return skew symmetric matrix used for the cross product of v with another vector.

    np.cross(v, w) == tilde(v) @ w.

    Parameters
    ----------
    v : numpy.array
        Vector to be converted to a cross product matrix.

    Returns
    -------
    vTilde : numpy.array
        Skew symmetric matrix equivalent to v.
    """
    vTilde = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return vTilde
