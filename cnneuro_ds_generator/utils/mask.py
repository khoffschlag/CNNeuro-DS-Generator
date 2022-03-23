from cnneuro_ds_generator.utils.mri import Volume
import numpy as np


def roi_mask(loc, val, atlas_file):
    """
    Creates mask where the in loc specified locations are set equal to val and every other location has the value 1.
    :param list/int/float loc: location(s) that should get set to the specified value
    :param int/float val: value for the location(s)
    :param str atlas_file: atlas file
    :return: mask
    """
    if not isinstance(loc, list) and not isinstance(loc, int) and not isinstance(loc, float):
        raise ValueError('loc must be int, float or a list!')
    if isinstance(loc, list) and any([not isinstance(elem, int) and not isinstance(elem, float) for elem in loc]):
        raise ValueError('Every element in loc has to be either a int or float!')
    if isinstance(loc, int) or isinstance(loc, float):
        loc = [loc]
    atlas = Volume(atlas_file)
    mask = np.copy(atlas.data)
    target = np.isin(mask, loc)
    mask[target] = val
    mask[np.logical_not(target)] = 1
    return mask


def concat(mri_data, mask_data):
    """
    Concat mri volume with a suitable mask
    :param numpy.ndarray mri_data: mri data array
    :param numpy.ndarray mask_data: mask data array
    :return: mri volume with applied mask
    """
    if not isinstance(mri_data, np.ndarray) or not isinstance(mask_data, np.ndarray):
        raise ValueError('mri_data and mask_data must be of type numpy.ndarray')
    if mri_data.shape != mask_data.shape:
        raise ValueError('Shape of the mask must be the same as the shape of the mri')
    return np.multiply(mri_data, mask_data)


def rand_loc_subset(loc, loc_max, seed):
    if loc_max > len(loc):
        raise ValueError('loc_max can not be greater than the amount of locations in loc!')
    np.random.seed(seed)
    loc = loc.copy()
    np.random.shuffle(loc)  # shuffle list to ensure that we take random locations
    amount_loc = np.random.randint(1, loc_max+1)  # draw the max amount of locations that is going to be used
    return loc[:amount_loc]
