"""
Default configurations for multi-site fMRI data classification
"""

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "./data"
_C.DATASET.QC = False  # Quality control, a parameter used for downlaod data
_C.DATASET.DOWNLOAD = True
_C.DATASET.BASE_DIR = 'ABIDE_pcp/cpac/filt_noglobal/'
_C.DATASET.ATLAS = "cc200"
_C.DATASET.PHENO_FILE = 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
_C.DATASET.PIPELINE = "cpac"
_C.DATASET.PHENO_ONLY = False  # whether use phenotype data only for classification
# ---------------------------------------------------------------------------- #
# ML METHOD SETUP
# ---------------------------------------------------------------------------- #
_C.METHOD = CN()
_C.METHOD.MODEL = "MIDA"
_C.METHOD.KHSIC = True
_C.METHOD.SEED = 1234
_C.METHOD.CONNECTIVITY = "TPE"
_C.METHOD.ALGORITHM = "Ridge"
_C.METHOD.LOSO = True  # Leave one site out cross validation, perform k-fold if False
_C.METHOD.ENSEMBLE = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "./data"  # output_dir
_C.OUTPUT.OUT_PATH = "./data/abide_tpe_mida_out/"
_C.OUTPUT.SAVE_FEATURE = True
_C.OUTPUT.OUT_FILE = "TPE"


def get_cfg_defaults():
    return _C.clone()
