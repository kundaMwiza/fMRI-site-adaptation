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
_C.DATASET.ROOT = "/media/shuo/MyDrive/data/brain"
_C.DATASET.QC = False
_C.DATASET.DOWNLOAD = True
_C.DATASET.BASE_DIR = 'ABIDE_pcp/cpac/filt_noglobal/'
_C.DATASET.ATLAS = "cc200"
_C.DATASET.PHENO_FILE = 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
_C.DATASET.PIPELINE = "cpac"
# ---------------------------------------------------------------------------- #
# Image processing
# ---------------------------------------------------------------------------- #
_C.PROC = CN()
_C.PROC.SCALE = 2

# ---------------------------------------------------------------------------- #
# Visualization
# ---------------------------------------------------------------------------- #
_C.METHOD = CN()
_C.METHOD.MODEL = "MIDA"
_C.METHOD.KHSIC = True
_C.METHOD.SEED = 1234
_C.METHOD.CONNECTIVITY = "TPE"
_C.METHOD.ALGORITHM = "Ridge"
_C.METHOD.LOVO = True
_C.METHOD.ENSEMBLE = False

# ---------------------------------------------------------------------------- #
# Machine learning pipeline
# ---------------------------------------------------------------------------- #
_C.PIPELINE = CN()
_C.PIPELINE.CLASSIFIER = "linear_svc"  # ["svc", "linear_svc", "lr"]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "/media/shuo/MyDrive/data/brain"  # output_dir
_C.OUTPUT.SAVE_FEATURE = True
_C.OUTPUT.SAVE_PATH = "/media/shuo/MyDrive/data/brain/ABIDE_pcp/cpac/"
_C.OUTPUT.OUT_FILE = ""


def get_cfg_defaults():
    return _C.clone()
