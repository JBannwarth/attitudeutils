""" Functions to use modified Rodrigues parameters (MRPs).
MRPs are represented as numpy vectors of three elements.
Written by: Jeremie X. J. Bannwarth
"""

import numpy as np
from attitudeutils.quaternion import dcm2quat
from attitudeutils.common import tilde


def mrpshort(sigma):
    """Ensure the MRPs represent the short rotation.

    Parameters
    ----------
    sigma : numpy.array
        MRPs representing either long or short rotation.

    Returns
    -------
    sigmaShort : numpy.array
        MRPs representing the short rotation.
    """
    if np.linalg.norm(sigma) > 1.0:
        sigmaShort = -sigma / (np.linalg.norm(sigma) ** 2)
    else:
        sigmaShort = sigma

    return sigmaShort


def mrpdiff(sigma, w):
    """Calculate the MRP derivative based on the angular velocity.

    Parameters
    ----------
    sigma : numpy.array
        MRPs representing the current orientation.
    w : numpy.array
        Current angular velocity vector in rad/s.

    Returns
    -------
    sigmaDot : numpy.array
        MRP derivative.
    """
    sigmaDot = 0.25 * mrpB(sigma) @ w
    return sigmaDot


def mrpB(sigma):
    """Calculate the B matrix used for computing the MRP derivatives.

    Parameters
    ----------
    sigma : numpy.array
        MRPs representing the current orientation.

    Returns
    -------
    B : numpy.array
        B matrix in the equation sigmaDot = (1/4) * B @ w used for calculating
        the MRP derivatives based on the body angular velocity w.
    """
    B = (
        (1 - np.linalg.norm(sigma) ** 2) * np.eye(3)
        + 2 * tilde(sigma)
        + 2 * np.outer(sigma, sigma)
    )
    return B


def mrpadd(sigmaAC, sigmaCB):
    """Add the rotations defined by two MRP sets.

    Perform the rotation [AB] = [AC][CB]

    Parameters
    ----------
    sigmaAC : numpy.array
        MRPs defining the second rotation.
    sigmaCB : numpy.array
        MRPs defining the first rotation.

    Returns
    -------
    sigmaAB : numpy.array
        MRPs defining the total rotation.
    """
    # Check for rotations close to 360 degrees (denominator close to zero)
    denominator = (
        1
        + np.linalg.norm(sigmaCB) ** 2 * np.linalg.norm(sigmaAC) ** 2
        - 2 * np.dot(sigmaCB, sigmaAC)
    )

    if np.abs(denominator) < 1e-6:
        sigmaCB = mrp2shadow(sigmaCB)

    sigmaCBNorm = np.linalg.norm(sigmaCB)
    sigmaACNorm = np.linalg.norm(sigmaAC)

    sigmaAB = (
        (1 - sigmaCBNorm**2) * sigmaAC
        + (1 - sigmaACNorm**2) * sigmaCB
        - 2 * np.cross(sigmaAC, sigmaCB)
    ) / (1 + sigmaCBNorm**2 * sigmaACNorm**2 - 2 * np.dot(sigmaCB, sigmaAC))

    if np.linalg.norm(sigmaAB) > 1:
        sigmaAB = mrp2shadow(sigmaAB)

    return sigmaAB


def mrpsub(sigmaAC, sigmaBC):
    """Subtract the rotations defined by two MRP sets.

    Perform the rotation [AB] = [AC]*inv([BC])

    Parameters
    ----------
    sigmaAC : numpy.array
        MRPs defining the rotation to be subtracted from.
    sigmaBC : numpy.array
        MRPs defining the rotation to subtract.

    Returns
    -------
    sigmaAB : numpy.array
        MRPs defining the resulting rotation.
    """
    # Check for rotations close to 360 degrees (denominator close to zero)
    denominator = (
        1
        + np.linalg.norm(sigmaBC) ** 2 * np.linalg.norm(sigmaAC) ** 2
        + 2 * np.dot(sigmaBC, sigmaAC)
    )

    if np.abs(denominator) < 1e-6:
        sigmaBC = mrp2shadow(sigmaBC)

    sigmaBCNorm = np.linalg.norm(sigmaBC)
    sigmaACNorm = np.linalg.norm(sigmaAC)

    sigmaAB = (
        (1 - sigmaBCNorm**2) * sigmaAC
        - (1 - sigmaACNorm**2) * sigmaBC
        + 2 * np.cross(sigmaAC, sigmaBC)
    ) / (1 + sigmaBCNorm**2 * sigmaACNorm**2 + 2 * np.dot(sigmaBC, sigmaAC))

    if np.linalg.norm(sigmaAB) > 1:
        sigmaAB = mrp2shadow(sigmaAB)

    return sigmaAB


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
        (1 - sigmaNorm**2) / (1 + sigmaNorm**2),
        2 * sigma / (1 + np.square(sigma)),
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
    sigmaTilde = tilde(sigma)
    sigmaNorm = np.linalg.norm(sigma)
    dcm = (
        np.eye(3)
        + (8.0 * np.dot(sigmaTilde, sigmaTilde) - 4 * (1 - sigmaNorm**2) * sigmaTilde)
        / (1.0 + sigmaNorm**2) ** 2
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
