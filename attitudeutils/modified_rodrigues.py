""" Functions to use modified Rodrigues parameters (MRPs).
MRPs are represented as numpy vectors of three elements.
Written by: Jeremie X. J. Bannwarth
"""

import numpy as np


def mrp2shadow(sigma):
    """Convert MRPs to MRPs shadow set.
    
    Parameters
    ----------
    sigma : numpy.array
        Modified Rodrigues parameters.

    Returns
    -------
    sigmaS : numpy.array
        Corresponding shadow set of modified Rodrigues parameters.
    """
    return -sigma / np.linalg.norm(sigma) ** 2


def mrp2quat(sigma):
    """Convert MRPs to quaternion.

    Parameters
    ----------
    sigma : numpy.array
        Modified Rodrigues parameters.

    Returns
    -------
    beta : numpy.array
        Corresponding quaternion.
    """
    sigmaNorm = np.linalg.norm(sigma)
    return np.append(
        (1 - sigmaNorm ** 2) / (1 + sigmaNorm ** 2), 2 * sigma / (1 + np.square(sigma)),
    )


def quat2mrp(beta):
    """Convert quaternion to MRPs.

    Parameters
    ----------
    beta : numpy.array
        Quaternion.

    Returns
    -------
    sigma : numpy.array
        Corresponding modified Rodrigues parameters.
    """
    return beta[1:] / (1 + beta[0])


def dcm2mrp(C):
    """Convert DCM to MRPs.

    Parameters
    ----------
    dcm : numpy.array
        Direction cosine matrix.

    Returns
    -------
    q : numpy.array
        Corresponding modified Rodrigues parameters.
    """
    zeta = np.sqrt(np.trace(C) + 1)
    sigma = np.array(
        [[C[1, 2] - C[2, 1]], [C[2, 0] - C[0, 2]], [C[0, 1] - C[1, 0]]]
    ) / (zeta * (zeta + 2))

    return sigma.ravel()
