""" Functions to use principal rotation vectors (PRV).
Written by: Jeremie X. J. Bannwarth
"""

import numpy as np


def prv2dcm(gamma):
    """Convert PRV to DCM.

    Parameters
    ----------
    sigma : numpy.array
        Principal rotation vector.

    Returns
    -------
    C : numpy.array
        Corresponding direction cosine matrix.
    """
    Phi = np.linalg.norm(gamma)
    e = gamma / Phi

    cPhi = np.cos(Phi)
    sPhi = np.sin(Phi)
    Sigma = 1 - cPhi

    return np.array(
        [
            [
                e[0] ** 2 * Sigma + cPhi,
                e[0] * e[1] * Sigma + e[2] * sPhi,
                e[0] * e[2] * Sigma - e[1] * sPhi,
            ],
            [
                e[1] * e[0] * Sigma - e[2] * sPhi,
                e[1] ** 2 * Sigma + cPhi,
                e[1] * e[2] * Sigma + e[0] * sPhi,
            ],
            [
                e[2] * e[0] * Sigma + e[1] * sPhi,
                e[2] * e[1] * Sigma - e[0] * sPhi,
                e[2] ** 2 * Sigma + cPhi,
            ],
        ]
    )


def dcm2prv(C):
    """ Convert DCM to PRV.

    Parameters
    ----------
    C : numpy.array
        Direction cosine matrix.

    Returns
    -------
    gamma : numpy.array
        Corresponding principal rotation vector.
    """
    Phi = np.arccos(0.5 * (C[0, 0] + C[1, 1] + C[2, 2] - 1))
    eHat = np.array([C[1, 2] - C[2, 1], C[2, 0] - C[0, 2], C[0, 1] - C[1, 0]]) / (
        2 * np.sin(Phi)
    )

    return Phi * eHat
