""" Functions to use classical rodrigues parameters (CRPs).
CRPs are represented as numpy vectors of three elements.
Written by: Jeremie X. J. Bannwarth
"""

import numpy as np


def crpdiff(q, w):
    """Calculate the CRP derivative based on the angular velocity.

    Parameters
    ----------
    q : numpy.array
        CRP representing the current orientation.
    p : numpy.array
        Current angular velocity vector in rad/s.

    Returns
    -------
    qDot : numpy.array
        CRP derivative.
    """
    B = np.array(
        [
            [1 + q[0] ** 2, q[0] * q[1] - q[2], q[0] * q[2] + q[1]],
            [q[1] * q[0] + q[2], 1 + q[1] ** 2, q[1] * q[2] - q[0]],
            [q[2] * q[0] - q[1], q[2] * q[1] + q[0], 1 + q[2] ** 2],
        ]
    )

    qDot = 0.5 * np.dot(B, w)
    return qDot


def crpadd(p, q):
    """ Add the rotations defined by two CRP sets.

    Perform the rotation [FN(qTot)] = [FB(p)][BN(q)]

    Parameters
    ----------
    p : numpy.array
        CRPs defining the second rotation.
    q : numpy.array
        CRPs defining the first rotation.

    Returns
    -------
    qTot : numpy.array
        CRPs defining the total rotation.
    """
    return (p + q - np.cross(p, q)) / (1 - np.dot(p, q))


def crp2dcm(q):
    """Convert CRPs to a DCM.

    Parameters
    ----------
    q : numpy.array
        Classical rodrigues parameters.

    Returns
    -------
    dcm : numpy.array
        Corresponding direction cosine matrix.
    """
    dcm = (1 / (1 + np.dot(q.T, q))) * np.array(
        [
            [
                1 + q[0] ** 2 - q[1] ** 2 - q[2] ** 2,
                2 * (q[0] * q[1] + q[2]),
                2 * (q[0] * q[2] - q[1]),
            ],
            [
                2 * (q[1] * q[0] - q[2]),
                1 - q[0] ** 2 + q[1] ** 2 - q[2] ** 2,
                2 * (q[1] * q[2] + q[0]),
            ],
            [
                2 * (q[2] * q[0] + q[1]),
                2 * (q[2] * q[1] - q[0]),
                1 - q[0] ** 2 - q[1] ** 2 + q[2] ** 2,
            ],
        ]
    )

    return dcm


def dcm2crp(C):
    """Convert DCM to CRPs.

    Parameters
    ----------
    dcm : numpy.array
        Direction cosine matrix.

    Returns
    -------
    q : numpy.array
        Corresponding classical rodrigues parameters.
    """
    zeta = np.sqrt(np.trace(C) + 1)
    q = (
        np.array([[C[1, 2] - C[2, 1]], [C[2, 0] - C[0, 2]], [C[0, 1] - C[1, 0]]])
        / zeta ** 2
    )

    return q.ravel()
