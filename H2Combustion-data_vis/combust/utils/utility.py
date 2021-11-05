import numpy as np


def euler_rotation_matrix(theta):
    """
    Rotate the xyz values based on the euler angles, theta. Directly copied from:
    Credit:"https://www.learnopencv.com/rotation-matrix-to-euler-angles/"

    Parameters
    ----------
    theta: numpy array
        A 1D array of angles along x, y and z directions

    Returns
    -------
    numpy array: rotation matrix with shape (3,3)
    """

    R_x = np.array([[1, 0, 0], [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]),
                     np.cos(theta[0])]])
    R_y = np.array([[np.cos(theta[1]), 0,
                     np.sin(theta[1])], [0, 1, 0],
                    [-np.sin(theta[1]), 0,
                     np.cos(theta[1])]])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0], [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def rotate_molecule(atoms, theta=None):
    """
    Rotates the structure of molecule between -pi/2 and pi/2.

    Parameters
    ----------
    atoms: numpy array
        An array of atomic positions with last dimension = 3

    theta: numpy array, optional (default: None)
        A 1D array of angles along x, y and z directions.
        If None, it will be generated uniformly between -pi/2 and pi/2

    Returns
    -------
    numpy array: The rotated atomic positions with shape (... , 3)

    """

    # handle theta
    if theta is None:
        theta = np.random.uniform(-np.pi / 2., np.pi / 2., size=3)

    # rotation matrix
    R = euler_rotation_matrix(theta)

    return np.dot(atoms, R)


def padaxis(array, new_size, axis, pad_value=0, pad_right=True):
    """
    Padds one axis of an array to a new size
    This is just a wrapper for np.pad, more usefull when only padding a single axis

    Parameters
    ----------
    array: ndarray
        the array to pad

    new_size: int
        the new size of the specified axis

    axis: int
        axis along which to pad

    pad_value: float or int, optional(default=0)
        pad value

    pad_right: bool, optional(default=True)
        if True pad on the right side, otherwise pad on left side

    Returns
    -------
    ndarray: padded array

    """
    add_size = new_size - array.shape[axis]
    assert add_size >= 0, 'Cannot pad dimension {0} of size {1} to smaller size {2}'.format(
        axis, array.shape[axis], new_size)
    pad_width = [(0, 0)] * len(array.shape)

    #pad after if int is provided
    if pad_right:
        pad_width[axis] = (0, add_size)
    else:
        pad_width[axis] = (add_size, 0)

    return np.pad(array,
                  pad_width=pad_width,
                  mode='constant',
                  constant_values=pad_value)
