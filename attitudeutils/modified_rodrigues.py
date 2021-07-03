""" Functions to use modified Rodrigues parameters (MRPs).
MRPs are represented as numpy vectors of three elements.
Written by: Jeremie X. J. Bannwarth
"""

import numpy as np
from attitudeutils.quaternion import dcm2quat


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


def mrp2dcm(sigma):
    """Convert MRPs to DCM.

    Parameters
    ----------
    sigma : numpy.array
        Corresponding modified Rodrigues parameters.

    Returns
    -------
    dcm : numpy.array
        Direction cosine matrix.
    """
    sigmaTilde = np.array(
        [[0, -sigma[2], sigma[1]], [sigma[2], 0, -sigma[0]], [-sigma[1], sigma[0], 0]]
    )
    sigmaNorm = np.linalg.norm(sigma)
    dcm = (
        np.eye(3)
        + (8.0 * np.dot(sigmaTilde, sigmaTilde) - 4 * (1 - sigmaNorm ** 2) * sigmaTilde)
        / (1.0 + sigmaNorm ** 2) ** 2
    )

    return dcm


def dcm2mrp(C):
    """Convert DCM to MRPs.

    Convert to quaternion first to avoid singularity.

    Parameters
    ----------
    dcm : numpy.array
        Direction cosine matrix.

    Returns
    -------
    sigma : numpy.array
        Corresponding modified Rodrigues parameters.
    """
    beta = dcm2quat(C)

    return quat2mrp(beta)
