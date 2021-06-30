""" Functions to use quaternions.
Quaternions are represented as numpy vectors of four elements.
Written by: Jeremie X. J. Bannwarth
"""

import numpy as np


def quat2dcm(q):
    """Convert quaternion to DCM.

    Parameters
    ----------
    q : numpy.array
        Quaternion in the form [q0 q1 q2 q3], where q0 is the rotation element.

    Returns
    -------
    dcm : numpy.array
        Corresponding direction cosine matrix.
    """
    return np.array(
        [
            [
                q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
                2 * (q[1] * q[2] + q[0] * q[3]),
                2 * (q[1] * q[3] - q[0] * q[2]),
            ],
            [
                2 * (q[1] * q[2] - q[0] * q[3]),
                q[0] ** 2 - q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
                2 * (q[2] * q[3] + q[0] * q[1]),
            ],
            [
                2 * (q[1] * q[3] + q[0] * q[2]),
                2 * (q[2] * q[3] - q[0] * q[1]),
                q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2,
            ],
        ]
    )


def dcm2quat(C):
    """ Convert DCM to quaternion.

    Using Sheppard's method.

    Parameters
    ----------
    C : numpy.array
        Direction cosine matrix.

    Returns
    -------
    q : numpy.array
        Corresponding quaternion.
    """
    tr = np.trace(C)
    qq = 0.25 * np.array(
        [
            1.0 + tr,
            1.0 + 2.0 * C[0, 0] - tr,
            1.0 + 2.0 * C[1, 1] - tr,
            1 + 2.0 * C[2, 2] - tr,
        ]
    )

    idxMax = np.argmax(qq)
    if idxMax == 0:
        q = 0.25 * np.array(
            [4 * qq[0], C[1, 2] - C[2, 1], C[2, 0] - C[0, 2], C[0, 1] - C[1, 0]]
        )
    elif idxMax == 1:
        q = 0.25 * np.array(
            [C[1, 2] - C[2, 1], 4 * qq[1], C[0, 1] + C[1, 0], C[2, 0] + C[0, 2]]
        )
    elif idxMax == 2:
        q = 0.25 * np.array(
            [C[2, 0] - C[0, 2], C[0, 1] + C[1, 0], 4 * qq[2], C[1, 2] + C[2, 1]]
        )
    else:
        q = 0.25 * np.array(
            [C[0, 1] - C[1, 0], C[2, 0] + C[0, 2], C[1, 2] + C[2, 1], 4 * qq[3]]
        )

    q = q / np.sqrt(qq[idxMax])

    if q[0] < 0:
        q = -q

    return q
