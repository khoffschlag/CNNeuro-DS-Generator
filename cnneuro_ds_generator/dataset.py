import functools
import glob
import random
import os
import shutil
import nibabel as nib
import pandas as pd
import cnneuro_ds_generator

def get_ixi_id(ixi_file):
    """ Takes path to ixi subject and returns a string containing the ixi id """
    return ixi_file.split('/')[-1].split('IXI')[-1][0:3]


def __gather_subs(input_dir, amount_ill_patients, amount_healthy_patients, shuffle, seed):
    patients = glob.glob(input_dir + '/**/*.nii', recursive=True)
    patients.extend(glob.glob(input_dir + '/**/*.nii.gz', recursive=True))
    patients = sorted(patients, key=lambda file: int(get_ixi_id(file)))

    print('Found', len(patients), 'nifti files.')
    if amount_ill_patients > len(patients):
        raise ValueError (
            'You can have less patients in the resulting dataset than in the original dataset but not more!'
        )
    random.seed(seed)
    if shuffle:
        random.shuffle(patients)
    ill_patients = patients[:amount_ill_patients]
    healthy_patients = patients[amount_ill_patients:amount_ill_patients+amount_healthy_patients]
    return ill_patients, healthy_patients


def __save_sub(mri_file, subject_id, output_dir, ill):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if ill:
        dest = os.path.join(output_dir, 'sub-' + subject_id + '-ill.nii')
        nib.save(mri_file, dest)
    else:
        dest = os.path.join(output_dir, 'sub-' + subject_id + '-healthy.nii')
        shutil.copy(mri_file, dest)


def __generate_tumor_demographic(output_dir, ill_subs, healthy_subs):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    data = {'id': [], 'ill': []}

    for ill_sub in ill_subs:
        data['id'].append(get_ixi_id(ill_sub))
        data['ill'].append(1)

    for healthy_sub in healthy_subs:
        data['id'].append(get_ixi_id(healthy_sub))
        data['ill'].append(0)

    df = pd.DataFrame(data, columns=['id', 'ill'])
    df.to_csv(os.path.join(output_dir, 'demographic.csv'))


def generate_dataset(input_dir, output_dir, amount_healthy_subs, amount_ill_subs, transformer, transformer_settings,
                     seed, shuffle_subjects=False):
    input_dir = os.path.abspath(input_dir)
    if not os.path.exists(input_dir):
        raise Exception('The specified input_dir %s does not exist!' % input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ill_subs, healthy_subs = __gather_subs(input_dir=input_dir, amount_ill_patients=amount_ill_subs,
                                           amount_healthy_patients=amount_healthy_subs, shuffle=shuffle_subjects,
                                           seed=seed)

    transformer_func = functools.partial(transformer, **transformer_settings)
    current_seed = seed
    for ill_sub in ill_subs:
        transformer_func = functools.partial(transformer_func, **{'mri_file': ill_sub})
        transformer_func = functools.partial(transformer_func, **{'seed': current_seed})
        new_mri = transformer_func()
        current_seed += 1
        __save_sub(mri_file=new_mri, subject_id=get_ixi_id(ill_sub), output_dir=output_dir, ill=True)

    for healthy_sub in healthy_subs:
        __save_sub(mri_file=healthy_sub, subject_id=get_ixi_id(healthy_sub), output_dir=output_dir, ill=False)

    __generate_tumor_demographic(output_dir=output_dir, ill_subs=ill_subs, healthy_subs=healthy_subs)
    version_file = os.path.join(output_dir, 'version.txt')
    version = cnneuro_ds_generator.__version__
    with open(version_file, 'w') as file:
        file.write('This dataset was created with version %s of the CNNeuro-DS-Generator.' % version)

