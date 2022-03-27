# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, , Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from nilearn import datasets
# import argparse
from imports import preprocess_data as reader
# from imports.utils import str2bool
import os
import shutil
# import sys
from imports.utils import arg_parse
from config import get_cfg_defaults

# Input data variables
# root_folder = '/path/to/data/'
# root_folder = "/media/shuo/MyDrive/data/brain"
# root_folder = "D:/ML_data/brain/qc"
# data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal/')


def main():
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    # parser = argparse.ArgumentParser(description='Download ABIDE data and compute functional connectivity matrices')
    # parser.add_argument('--pipeline', default='cpac', type=str,
    #                     help='Pipeline to preprocess ABIDE data. Available options are ccs, cpac, dparsf and niak. '
    #                          'Default: cpac.')
    # parser.add_argument('--atlas', default='cc200', help='Brain parcellation atlas. Options: ho, cc200 and cc400, '
    #                                                      'default: cc200.')
    # parser.add_argument('--connectivity', default='correlation', type=str,
    #                     help='Type of connectivity used for network construction options: correlation, '
    #                          'partial correlation, covariance, tangent, TPE. Default: correlation.')
    # parser.add_argument('--download', default=True, type=str2bool,
    #                     help='Dowload data or just compute functional connectivity. default: True')
    # args = parser.parse_args()
    # print(args)

    # params = dict()

    # pipeline = args.pipeline
    # atlas = args.atlas
    # connectivity = args.connectivity
    # download = args.download

    pipeline = cfg.DATASET.PIPELINE
    atlas = cfg.DATASET.ATLAS
    connectivity = cfg.METHOD.CONNECTIVITY
    download = cfg.DATASET.DOWNLOAD

    root_dir = cfg.DATASET.ROOT
    data_folder = os.path.join(root_dir, cfg.DATASET.BASE_DIR)

    # Files to fetch

    files = ['rois_' + atlas]

    # filemapping = {'func_preproc': 'func_preproc.nii.gz', files[0]: files[0] + '.1D'}

    # if not os.path.exists(data_folder):
    #     os.makedirs(data_folder)
    # shutil.copyfile('./subject_ids.txt', os.path.join(data_folder, 'subject_ids.txt'))

    # Download database files
    phenotype_file = os.path.join(root_dir, "ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    if download or not os.path.exists(phenotype_file):
        datasets.fetch_abide_pcp(data_dir=root_dir, pipeline=pipeline, band_pass_filtering=True,
                                 global_signal_regression=False, derivatives=files, quality_checked=cfg.DATASET.QC)

    phenotype_df = reader.get_phenotype(phenotype_file)
    # fnames = phenotype_df["FILE_ID"].values

    # subject_ids = reader.get_ids(data_folder)
    # subject_ids = subject_ids.tolist()
    subject_ids = []
    # Create a folder for each subject
    for i in phenotype_df.index:
        sub_id = phenotype_df.loc[i, "SUB_ID"]

        subject_folder = os.path.join(data_folder, "%s" % sub_id)
        if not os.path.exists(subject_folder):
            os.mkdir(subject_folder)
        for fl in files:
            fname = "%s_%s.1D" % (phenotype_df.loc[i, "FILE_ID"], fl)
            data_file = os.path.join(data_folder, fname)
            if os.path.exists(data_file) or os.path.exists(os.path.join(subject_folder, fname)):
                subject_ids.append(sub_id)
                if not os.path.exists(os.path.join(subject_folder, fname)):
                    shutil.move(data_file, subject_folder)

    sub_id_fpath = os.path.join(data_folder, "subject_ids.txt")
    if not os.path.exists(sub_id_fpath):
        f = open(sub_id_fpath, "w")
        for sub_id_ in subject_ids:
            f.write("%s\n" % sub_id_)
        f.close()
    else:
        subject_ids = reader.get_ids(data_folder)
        subject_ids = subject_ids.tolist()

    # for s, fname in zip(subject_ids, reader.fetch_filenames(subject_ids, files[0], atlas, data_folder)):
    #     subject_folder = os.path.join(data_folder, s)
    #     if not os.path.exists(subject_folder):
    #         os.mkdir(subject_folder)
    #
    #     # Get the base filename for each subject
    #     base = fname.split(files[0])[0]
    #
    #     # Move each subject file to the subject folder
    #     for fl in files:
    #         if not os.path.exists(os.path.join(subject_folder, base + filemapping[fl])):
    #             shutil.move(base + filemapping[fl], subject_folder)
    #
    time_series = reader.get_timeseries(subject_ids, atlas, data_folder)

    # Compute and save connectivity matrices
    # if connectivity in ['correlation', 'partial correlation', 'covariance']:
    if connectivity in ["correlation", 'partial correlation', 'covariance', 'tangent', "TPE"]:
        # reader.subject_connectivity(time_series, subject_ids, atlas, connectivity)
        reader.subject_connectivity(time_series, atlas, connectivity, save=True, out_path=cfg.OUTPUT.OUT_PATH)


if __name__ == '__main__':
    main()
