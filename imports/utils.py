
import argparse
import os


# Process boolean command line arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Input data variables
root_dir_default = "D:/ML_data/brain/qc"
data_folder_name_default = 'ABIDE_pcp/cpac/filt_noglobal/'
pheno_fname_default = 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'


class DataPath:
    def __init__(self, root_dir=None, data_folder_name=None, pheno_fname=None):
        if root_dir is None:
            root_dir = root_dir_default
        if data_folder_name is None:
            data_folder_name = data_folder_name_default
        if pheno_fname is None:
            pheno_fname = pheno_fname_default
        self._root_dir = root_dir
        self._data_folder = os.path.join(root_dir, data_folder_name)
        self._pheno_fpath = os.path.join(root_dir, pheno_fname)

    def get_root_dir(self):
        return self._root_dir

    def get_data_folder(self):
        return self._data_folder

    def get_pheno_fpath(self):
        return self._pheno_fpath
