# Copyright (c) 2019 Mwiza Kunda
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


# import sys
# import argparse
# import pandas as pd
import os
import numpy as np
from imports import preprocess_data as reader
from imports import train as train
# from imports.utils import str2bool
import warnings
from imports.utils import arg_parse
from config import get_cfg_defaults

warnings.filterwarnings("ignore")


def main():
    # parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a Ridge classifier. MIDA '
    #                                              'is used to minimize the distribution mismatch between ABIDE sites')
    # parser.add_argument('--atlas', default='cc200', help='Atlas for network construction (node definition) options: '
    #                                                      'ho, cc200, cc400, default: cc200.')
    # parser.add_argument('--model', default='MIDA', type=str, help='Options: MIDA, raw. default: MIDA.')
    # parser.add_argument('--algorithm', default='Ridge', type=str, help='Options: Ridge, LR (Logistic regression), '
    #                                                                    'SVM (Support vector machine). default: Ridge.')
    # parser.add_argument('--phenotypes', default=True, type=str2bool, help='Add phenotype features. default: True.')
    # parser.add_argument('--KHSIC', default=True, type=str2bool, help='Compute kernel statistical test of independence '
    #                                                                  'between features and site, default True.')
    # parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation. default: 1234.')
    # parser.add_argument('--connectivity', default='TPE', type=str, help='Type of connectivity used for network '
    #                                                                     'construction. options: correlation, '
    #                                                                     'TE(tangent embedding), TPE(tangent pearson '
    #                                                                     'embedding), default: TPE.')
    # parser.add_argument('--leave_one_out', default=False, type=str2bool,
    #                     help='leave one site out CV instead of 10CV. Default: False.')
    # parser.add_argument('--filename', default='tangent', type=str, help='filename for output file. default: tangent.')
    # parser.add_argument('--ensemble', default=False, type=str2bool, help='Leave one site out, use ensemble MIDA/raw. '
    #                                                                      'Default: False')
    #
    # args = parser.parse_args()
    # print('Arguments: \n', args)
    #

    args = arg_parse()
    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    # # Ridge classifier parameters
    params = dict()
    params['model'] = cfg.METHOD.MODEL  # MIDA, SMIDA or raw
    params['phenotypes'] = cfg.DATASET.PHENO_FILE  # Add phenotype features
    params['seed'] = cfg.METHOD.SEED  # seed for random initialisation
    params['ensemble'] = cfg.METHOD.ENSEMBLE

    # Algorithm choice
    params['algorithm'] = cfg.METHOD.ALGORITHM
    # Compute kernel statistical test of independence between features and site (boolean)
    params['KHSIC'] = cfg.METHOD.KHSIC
    params['filename'] = cfg.OUTPUT.OUT_FILE  # Results output file
    params['connectivity'] = cfg.METHOD.CONNECTIVITY  # Type of connectivity used for network construction
    params['atlas'] = cfg.DATASET.ATLAS  # Atlas for network construction

    atlas = cfg.DATASET.ATLAS  # Atlas for network construction (node definition)
    connectivity = cfg.METHOD.CONNECTIVITY  # Type of connectivity used for network construction

    root_dir = cfg.DATASET.ROOT
    data_folder = os.path.join(root_dir, cfg.DATASET.BASE_DIR)
    pheno_fpath = os.path.join(root_dir, cfg.DATASET.PHENO_FILE)
    params["data_path"] = data_folder
    params["pheno_fpath"] = pheno_fpath

    # 10 Fold CV or leave one site out CV
    params['leave_one_out'] = cfg.METHOD.LOVO
    if params['leave_one_out']:
        params['validation_ext'] = 'LOCV'
    else:
        params['validation_ext'] = '10CV'

    # Get subject IDs and class labels
    subject_ids = reader.get_ids(data_folder)
    labels = reader.get_subject_score(subject_ids, score='DX_GROUP', pheno_fpath=pheno_fpath)

    # Number of subjects and classes for binary classification
    n_classes = 2
    n_subjects = len(subject_ids)
    params['n_subs'] = n_subjects

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([n_subjects, n_classes])
    y = np.zeros([n_subjects, 1])

    # Get class labels for all subjects
    for i in range(n_subjects):
        y_data[i, int(labels[subject_ids[i]]) - 1] = 1
        y[i] = int(labels[subject_ids[i]])

    # Compute feature vectors (vectorised connectivity networks)
    # if connectivity not in ['tangent', 'TPE']:
    #     features = reader.get_networks(subject_ids, kind=connectivity, data_path=data_folder, iter_no='', atlas=atlas)
    # else:
    #     features = None
    features = reader.get_networks(subject_ids, kind=connectivity, data_path=data_folder, iter_no='', atlas=atlas)

    # Source phenotype information and preprocess phenotypes
    if cfg.METHOD.MODEL == 'MIDA':
        pheno_ft = reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID', 'HANDEDNESS_CATEGORY', 'AGE_AT_SCAN',
                                                             'FIQ', 'VIQ', 'PIQ'], subject_ids, pheno_fpath)
    else:
        pheno_ft = reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID', 'EYE_STATUS_AT_SCAN',
                                                             'HANDEDNESS_CATEGORY', 'AGE_AT_SCAN', 'FIQ', 'VIQ', 'PIQ'],
                                                            subject_ids, pheno_fpath)

    pheno_ft.index = subject_ids
    pheno_ft2 = pheno_ft

    # number of sites available in the dataset
    params['n_domains'] = len(pheno_ft2['SITE_ID'].unique())

    # preprocess categorical data ordinally
    pheno_ft = reader.preprocess_phenotypes(pheno_ft, params)

    # le = LabelEncoder()
    # site_label = le.fit_transform(pheno_ft["SITE_ID"].values)
    # out_site_file = os.path.join(cfg.OUTPUT.OUT_PATH, "site_label.mat")
    # sio.savemat(out_site_file, {'site_label': site_label})

    # construct phenotype feature vectors
    phenotype_ft = reader.phenotype_ft_vector(pheno_ft, n_subjects, params)

    if params['leave_one_out']:
        # leave one site out evaluation
        if params['ensemble']:
            train.leave_one_site_out_ensemble(params, subject_ids, features, y_data, y, phenotype_ft, pheno_ft)
        else:
            train.leave_one_site_out(params, subject_ids, features, y_data, y, phenotype_ft, pheno_ft)
    else:
        # 10 fold CV evaluation
        train.train_10CV(params, subject_ids, features, y_data, y, phenotype_ft, pheno_ft)


if __name__ == '__main__':
    main()
