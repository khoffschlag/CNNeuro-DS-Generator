from generator.utils.mri import Volume, gaussian_blur
from generator.utils.mask import roi_mask, concat, rand_loc_subset
import numpy as np


def __apply_decay(val, decay_val, lower_bound, upper_bound):
    if val < lower_bound or val > upper_bound:
        raise ValueError('the value which should be decayed should be between lower bound and upper bound!')
    x = val - decay_val
    if lower_bound <= x <= upper_bound:
        return x
    else:
        return val


def __apply_offset(pos, offset_range, seed):
    min_offset = offset_range[0]
    max_offset = offset_range[1]
    pos_x, pos_y, pos_z = pos
    np.random.seed(seed)
    pos_x += np.random.randint(min_offset[0], max_offset[0] + 1)
    pos_y += np.random.randint(min_offset[1], max_offset[1] + 1)
    pos_z += np.random.randint(min_offset[2], max_offset[2] + 1)
    return pos_x, pos_y, pos_z


def atrophy(mri_file, atrophy_val, seed, atlas=None, loc=None ,loc_max=None, smoothing_sigma=False):
    """
    Add atrophy to mri
    :param str mri_file: volume to which atrophy should be added
    :param list/int/float loc: location(s) that should show atrophy
    :param int loc_max: if a list is passed as loc and loc_max is set, randomly between 1 and loc_max random locations
                        of this list are going to get atrophy. If loc_max is None, every specified element is used.
    :param tuple/int/float atrophy_val: the factor with which the location(s) will be multiplied. You can pass a tuple
                                        (min, max) and a value between min (included) and max (excluded) will be
                                        randomly determined. You can also pass an int or float ant this value will be
                                        used as the factor.
    :param bool/float smoothing_sigma: If not False, smoothing with given sigma is applied to the mask
    :param str atlas: path to atlas file
    :param int seed: RNG seed
    :return: nifti-Image with atrophy
    """
    if loc is not None:
        if not isinstance(loc, list) and not isinstance(loc, int) and not isinstance(loc, float):
            raise ValueError('loc must be int, float or a list!')
        if isinstance(loc, list) and any([not isinstance(elem, int) and not isinstance(elem, float) for elem in loc]):
            raise ValueError('Make sure that loc is either int, float or a list containing int or float elements!')
        if isinstance(loc, int) or isinstance(loc, float):
            loc = [loc]

    if loc_max is not None:  # user wants subset of all locations
        loc = rand_loc_subset(loc=loc, loc_max=loc_max, seed=seed)  # randomly take loc_max locations

    if isinstance(atrophy_val, tuple):  # user wants a random value x with atrophy_val[0] <= x < atrophy_val[1]
        min_val, max_val = atrophy_val
        np.random.seed(seed)
        atrophy_val = np.random.uniform(min_val, max_val)

    mri = Volume(mri_file=mri_file)
    if loc is not None:
        mask = roi_mask(loc=loc, val=atrophy_val, atlas_file=atlas)
        if smoothing_sigma:
            mask = gaussian_blur(data_arr=mask, sigma=smoothing_sigma)
        new_data = concat(mri.data, mask)
    else:
        new_data = mri.data * atrophy_val
    return mri.create_nifti(new_data)


def lesion(mri_file, change_prob, intensity, size, seed, size_offset=((0, 0, 0), (0, 0, 0)),
           pos_offset=((0, 0, 0), (0, 0, 0)), loc=None, atlas=None):
    """
    Add lesion to mri
    DO NOT SET mri_file AND seed THROUGH THE END-USER-TRANSFORMER SETTINGS
    If change_prob is a numeric value then this value is constantly used.
    change_prob can also be a tuple (start_val, decay, lower_bound, upper_bound) then we decay start_val with every
    voxel, but we stay between lower_bound and upper_bound).
    If intensity is a tuple (min, max) then a random value between min and max will be drawn.
    If intensity is a tuple (start_val, decay, lower_bound, upper_bound) then we decay start_val with every voxel, but
    we stay between lower_bound and upper_bound).
    intensity can also be 'mean' (use mean of mri) or if you supply a numeric value, this value will be constantly used!
    if loc is not None & not list -> put lesion in specified region
    if log is not None & list -> take a random location of the list
    :param mri_file: volume to which lesions should be added
    :param change_prob: the probability that a voxel is going to be changed
    :param loc: if None -> randomly around mid
    :param intensity: random, mean, fixed, decay
    :param size: tuple (x_size, y_size, z_size)
    :param size_offset: tuple ((min_offset_x, min_offset_y, min_offset_z), (max_offset_x, max_offset_y, max_offset_z))
    :param pos_offset: only used when in random mode. tuple ((min_offset_x, min_offset_y, min_offset_z), (max_offset_x, max_offset_y, max_offset_z))
    :param int seed: RNG seed
    :return:
    """

    if isinstance(intensity, int):
        intensity = float(intensity)

    volume = Volume(mri_file=mri_file)
    mri_data = volume.data

    if loc is not None:  # brain regions were specified
        if isinstance(loc, list):
            if len(loc) > 1:
                np.random.seed(seed)
                np.random.shuffle(loc)
            loc = loc[0]
        mask = roi_mask(loc=loc, val=2, atlas_file=atlas)  # create mask where the target areas are equal to 2 and every other area 1
        mask = mask - 1 # target areas become equal to 1 and non-target areas equal to 0
        is_target = np.nonzero(mask)  # true-false-array where target voxels equal 1
        # average(x), average(y), average(z)
        axis_1 = int(np.round(np.average(is_target[0])))
        axis_2 = int(np.round(np.average(is_target[1])))
        axis_3 = int(np.round(np.average(is_target[2])))
        start = (axis_1, axis_2, axis_3)
    else:
        start = __apply_offset(pos=volume.mid_pos(), offset_range=pos_offset, seed=seed)

    size = __apply_offset(pos=size, offset_range=size_offset, seed=seed)

    x_start, y_start, z_start = start
    x_end = x_start + size[0]
    y_end = y_start + size[1]
    z_end = z_start + size[2]

    current_seed = seed
    if isinstance(change_prob, tuple) and len(change_prob) == 4:
        change_prob_val, change_prob_decay, change_prob_lower_bound, change_prob_upper_bound = change_prob
    if isinstance(intensity, tuple) and len(intensity) == 4:
        intensity_val, intensity_decay, intensity_lower_bound, intensity_upper_bound = intensity

    for x_index in range(x_start, x_end + 1):
        for y_index in range(y_start, y_end + 1):
            for z_index in range(z_start, z_end + 1):
                np.random.seed(current_seed)
                current_seed += 1
                if np.random.random() < change_prob:
                    if isinstance(intensity, float):
                        mri_data[x_index, y_index, z_index] = intensity
                    elif isinstance(intensity, tuple) and len(intensity) == 2:
                        mri_data[x_index, y_index, z_index] = np.random.uniform(intensity[0], intensity[1])
                    elif isinstance(intensity, tuple) and len(intensity) == 4:
                        mri_data[x_index, y_index, z_index] = intensity_val
                        intensity_val = __apply_decay(
                            val=intensity_val,
                            decay_val=intensity_decay,
                            lower_bound=intensity_lower_bound,
                            upper_bound=intensity_upper_bound
                        )
                if isinstance(change_prob, tuple) and len(change_prob) == 4:
                    change_prob = __apply_decay(
                        val=change_prob_val,
                        decay_val=change_prob_decay,
                        lower_bound=change_prob_lower_bound,
                        upper_bound=change_prob_upper_bound
                    )
    return volume.create_nifti(mri_data)
