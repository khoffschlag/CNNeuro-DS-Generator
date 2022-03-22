from generator.dataset import generate_dataset
from generator.utils.transform import atrophy

INPUT_DIR = 'data_in/IXI-GM'
ATLAS_FILE = 'data_in/atlas/AAL3v1_1mm.nii.gz'
OUTPUT_DIR = 'data_out/Example_Output'

AMOUNT_HEALTHY = 50
AMOUNT_ILL = 50
SEED = 42

ROI = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 20]  # the frontal lobe

# Our IXI-GM-Dataset was smoothed with 8mm FWHM with SPM
# FWHM =  2.35482004503 * sigma
# 8 = 2.35482004503 * sigma -> sigma = 3.397287201153446
# I will use a sigma of 3.4 for smoothing

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
