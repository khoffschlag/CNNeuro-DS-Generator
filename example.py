from cnneuro_ds_generator.dataset import generate_dataset
from cnneuro_ds_generator.utils.transform import atrophy

INPUT_DIR = 'data_in/IXI-GM'  # make sure that this path leads to a suitable source dataset
ATLAS_FILE = 'data_in/atlas/AAL3v1_1mm.nii.gz'  # make sure that this path leads to a suitable brain atlas
OUTPUT_DIR = 'data_out/Example_Output'  # path to the folder where the generated files will take place

AMOUNT_HEALTHY = 50
AMOUNT_ILL = 50
SEED = 42

# Specify the brain regions (referenced according to the respective atlas) in which the changes are to be made:
ROI = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 20]

TRANSFORMER = atrophy
TRANSFORMER_SETTINGS = {
    'loc': ROI,
    'atrophy_val': 0.1,
    'atlas': ATLAS_FILE,
    'smoothing_sigma': 3.4
}

generate_dataset(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    amount_healthy_subs=AMOUNT_HEALTHY,
    amount_ill_subs=AMOUNT_ILL,
    shuffle_subjects=False,
    transformer=TRANSFORMER,
    transformer_settings=TRANSFORMER_SETTINGS,
    seed=SEED
)
