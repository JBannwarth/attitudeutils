""" Functions to use Euler angles.
Written by: Jeremie X. J. Bannwarth
"""

import numpy as np


def rotation1(theta):
    """Perform rotation about the 1-axis.
    
    Parameters
    ----------
    theta : double
        Angle of rotation in radians.

    Returns
    -------
    dcm : numpy.array
        Corresponding direction cosine matrix.
    """
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), np.sin(theta)],
            [0.0, -np.sin(theta), np.cos(theta)],
        ]
    )


def rotation2(theta):
    """Perform rotation about the 1-axis.
    
    Parameters
    ----------
    theta : double
        Angle of rotation in radians.

    Returns
    -------
    dcm : numpy.array
        Corresponding direction cosine matrix.
    """
    return np.array(
        [
            [np.cos(theta), 0.0, -np.sin(theta)],
            [0.0, 1.0, 0.0],
            [np.sin(theta), 0.0, np.cos(theta)],
        ]
    )


def rotation3(theta):
    """Perform rotation about the 1-axis.
    
    Parameters
    ----------
    theta : double
        Angle of rotation in radians.

    Returns
    -------
    dcm : numpy.array
        Corresponding direction cosine matrix.
    """
    return np.array(
        [
            [np.cos(theta), np.sin(theta), 0.0],
            [-np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def eul_angles2dcm(eul, sequence="321"):
    """Convert Euler angles to DCM.

    Parameters
    ----------
    eul : numpy.array
        Euler angles in the format [theta1 theta2 theta3]
        where theta1 represent the first rotation.
    sequence : string
        Euler rotation sequence. "321" corresponds to rotation around the 3-axis
        first, then rotation around the new 2-axis, then around the new 1-axis.

    Returns
    -------
    dcm : numpy.array
        Corresponding direction cosine matrix.
    """
    if sequence == "321":
        dcm = np.matmul(
            rotation1(eul[2]), np.matmul(rotation2(eul[1]), rotation3(eul[0]))
        )
    elif sequence == "123":
        dcm = np.matmul(
            rotation3(eul[2]), np.matmul(rotation2(eul[1]), rotation1(eul[0]))
        )
    elif sequence == "313":
        dcm = np.matmul(
            rotation3(eul[2]), np.matmul(rotation1(eul[1]), rotation3(eul[0]))
        )
    elif sequence == "323":
        dcm = np.matmul(
            rotation3(eul[2]), np.matmul(rotation2(eul[1]), rotation3(eul[0]))
        )
    else:
        dcm = np.nan

    return dcm
