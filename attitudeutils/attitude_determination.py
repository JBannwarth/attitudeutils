""" Functions to determine attitude from sensor measurements.
Written by: Jeremie X. J. Bannwarth
"""

import numpy as np


def triad(v1B, v2B, v1N, v2N):
    """ Implementation of the TRIAD method.

    Inputs are automatically renormalised.

    Parameters
    ----------
    v1B : numpy.array
        First observation vector in body frame, which is assumed to be the most accurate of the two.
    v2B : numpy.array
        Second observation vector in body frame, which is assumed to be the least accurate one.
    v1N : numpy.array
        First vector in the known inertial frame.
    v2N : numpy.array
        Second vector in the known inertial frame.

    Returns
    -------
    Cbar : numpy.array
        Estimated orientation as a direction cosine matrix.
    """
    # Normalisation
    v1B = v1B / np.linalg.norm(v1B)
    v2B = v2B / np.linalg.norm(v2B)
    v1N = v1N / np.linalg.norm(v1N)
    v2N = v2N / np.linalg.norm(v2N)

    # Body frame triad
    t1B = v1B
    t2B = np.cross(v1B, v2B)
    t2B = t2B / np.linalg.norm(t2B)

    t3B = np.cross(t1B, t2B)

    # Inertial triad
    t1N = v1N
    t2N = np.cross(v1N, v2N)
    t2N = t2N / np.linalg.norm(t2N)

    t3N = np.cross(t1N, t2N)

    # Attitude
    BbarT = np.column_stack((t1B, t2B, t3B))
    NT = np.column_stack((t1N, t2N, t3N))

    BbarN = np.dot(BbarT, NT.T)

    return BbarN
