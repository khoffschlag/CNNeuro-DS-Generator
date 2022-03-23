from cnneuro_ds_generator.dataset import generate_dataset
from cnneuro_ds_generator.utils.transform import atrophy
import os

INPUT_DIR = '../data_in/IXI-GM'
ATLAS_FILE = '../data_in/atlas/AAL3v1_1mm.nii.gz'
OUTPUT_DIR = '../data_out/'

AMOUNT_HEALTHY = 280
AMOUNT_ILL = 280
SEED = 42

DS_GRID = {
    'Frontal': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 20],
    'Hippocampus': [41, 42],
    'Global': None  # leads to a selection of the whole brain
}

ATROPHY_VALUES = [.35, .45, .55, .65, .75, .85, .95, 1]

for key in DS_GRID.keys():
    for index in range(len(ATROPHY_VALUES)):
        atrophy_value = ATROPHY_VALUES[index]
        ds_name = key + '_' + str(index+1)
        ds_path = os.path.join(OUTPUT_DIR, ds_name)
        print("--- Start creation of the dataset '%s' with atrophy value %s ---" % (ds_path, str(atrophy_value)))
        TRANSFORMER = atrophy
        TRANSFORMER_SETTINGS = {
            'loc': DS_GRID[key],
            'atrophy_val': atrophy_value,
            'atlas': ATLAS_FILE,
            'smoothing_sigma': 3.4
        }

        generate_dataset(
            input_dir=INPUT_DIR,
            output_dir=ds_path,
            amount_healthy_subs=AMOUNT_HEALTHY,
            amount_ill_subs=AMOUNT_ILL,
            shuffle_subjects=False,
            transformer=TRANSFORMER,
            transformer_settings=TRANSFORMER_SETTINGS,
            seed=SEED
        )
