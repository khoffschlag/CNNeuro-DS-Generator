from scipy.ndimage.filters import gaussian_filter
import nibabel as nib
import numpy as np
import os


def mid_pos(data_arr):
    """
    Return the position of the mid of the volume
    :param numpy.ndarray data_arr: the MRI data array
    :return: tuple containing position of the mid
    """
    if not isinstance(data_arr, np.ndarray):
        raise ValueError('data_arr has to be of type numpy.ndarray!')
    x_shape, y_shape, z_shape = data_arr.shape
    x_mid = (x_shape - 1) // 2
    y_mid = (y_shape - 1) // 2
    z_mid = (z_shape - 1) // 2
    # the minus 1 are necessary since we start from 0 but shape starts from 1
    return x_mid, y_mid, z_mid


def rand_pos_shift(pos, min_offset, max_offset, seed):
    """
    Shift a point (e.g. (0,4,2)) randomly with an offset between min_offset (e.g. (-3,-2,-1)) and max_offset
    (e.g. (1,2,3))
    :param tuple pos: tuple of length 3 that contains the position that should get shifted
    :param tuple min_offset: tuple of length 3 that contain the amount that the point AT LEAST should be shifted
    :param tuple max_offset: tuple of length 3 that contain the amount that the point AT MOST should be shifted
    :param int seed: seed for random-number-generator
    :return: shifted point
    """
    for param in [pos, min_offset, max_offset]:
        if not isinstance(param, tuple) or len(param) != 3:
            raise ValueError('pos, min_offset and max_offset has to be tuples of length 3!')

    # unpack values
    x_min_offset, y_min_offset, z_min_offset = min_offset
    x_max_offset, y_max_offset, z_max_offset = max_offset

    # draw offsets
    np.random.seed(seed)
    x_offset = np.random.randint(x_min_offset, x_max_offset)
    y_offset = np.random.randint(y_min_offset, y_max_offset)
    z_offset = np.random.randint(z_min_offset, z_max_offset)

    # calc shifted points
    x = pos[0] + x_offset
    y = pos[1] + y_offset
    z = pos[2] + z_offset
    return x, y, z


def gaussian_blur(data_arr, sigma):
    """
    Apply gaussian blur
    :param numpy.ndarray data_arr: the MRI data array
    :param float/tuple sigma: gauss-sigma
    :return: image with applied gaussian blur
    """
    if not isinstance(data_arr, np.ndarray):
        raise ValueError('data_arr has to be of type numpy.ndarray!!!')
    return gaussian_filter(data_arr, sigma=sigma)


class Volume:
    def __init__(self, mri_file):
        self.mri = nib.load(mri_file)
        self.data = self.mri.get_fdata()
        self.data = np.asarray(self.data)

    def mean(self):
        return np.mean(self.data)

    def mid_pos(self):
        return mid_pos(self.data)

    def create_nifti(self, data_arr, path=None):
        """
        Create new Nifti-Image based on a passed mri-data-array and the original (not passed) header.
        :param numpy.ndarray data_arr: the MRI data array
        :param str/None path: If not None, save volume at specified place. Do not forget the filename with extension!
        :return: Nifti1Image
        """
        nifti = nib.Nifti1Image(data_arr, self.mri.affine, self.mri.header)
        if path is not None:
            if not path.startswith('/'):
                path = os.path.join(os.getcwd(), path)
            nib.save(nifti, path)
        return nifti
