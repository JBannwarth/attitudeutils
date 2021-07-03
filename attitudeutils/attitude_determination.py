""" Functions to determine attitude from sensor measurements.
Written by: Jeremie X. J. Bannwarth
"""

import numpy as np
from attitudeutils.quaternion import quat2dcm


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


def devenportq(vB, vN, w):
    """ Implementation of Devenport's q-method.

    Inputs are automatically renormalised.

    Parameters
    ----------
    vB : numpy.array
        Nx3 matrix, each row representing an observation vector in body frame.
    vN : numpy.array
        Nx3 matrix, each row representing an vector in world frame.
    w : numpy.array
        Vector of length N of weights.

    Returns
    -------
        Cbar : numpy.array
        Estimated orientation as a direction cosine matrix.
    """
    # Calculate the B matrix
    N = w.shape[0]
    B = np.zeros((3, 3))
    for k in range(0, N):
        vB[k, :] = vB[k, :] / np.linalg.norm(vB[k, :])
        vN[k, :] = vN[k, :] / np.linalg.norm(vN[k, :])
        B = B + w[k] * np.outer(vB[k, :].ravel(), vN[k, :].ravel())

    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([[B[1, 2] - B[2, 1]], [B[2, 0] - B[0, 2]], [B[0, 1] - B[1, 0]]])

    # Calculate the K matrix
    Ktop = np.array([[sigma, Z[0, 0], Z[1, 0], Z[2, 0]]])
    Kbottom = np.hstack((Z, S - sigma * np.eye(3)))
    K = np.vstack((Ktop, Kbottom))

    # Find the largest eigenvalue and the corresponding eigenvector/quaternion
    lambdas, betas = np.linalg.eig(K)
    idx = np.argmax(lambdas)

    beta = betas[:, idx].ravel()

    if beta[0] < 0:
        beta = -beta

    Cbar = quat2dcm(beta)

    return Cbar
