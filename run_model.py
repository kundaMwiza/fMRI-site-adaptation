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


import sys
import argparse
import pandas as pd
import numpy as np
from imports import preprocess_data as reader
from imports import train as train
from imports.utils import str2bool
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a Ridge classifier. MIDA '
                                                 'is used to minimize the distribution mismatch between ABIDE sites')
    parser.add_argument('--atlas', default='cc200', help='Atlas for network construction (node definition) options: '
                                                         'ho, cc200, cc400, default: cc200.')
    parser.add_argument('--model', default='MIDA', type=str, help='Options: MIDA, raw. default: MIDA.')
    parser.add_argument('--algorithm', default='Ridge', type=str, help='Options: Ridge, LR (Logistic regression), '
                                                                       'SVM (Support vector machine). default: Ridge.')
    parser.add_argument('--phenotypes', default=True, type=str2bool, help='Add phenotype features. default: True.')
    parser.add_argument('--KHSIC', default=True, type=str2bool, help='Compute kernel statistical test of independence '
                                                                     'between features and site, default True.')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation. default: 1234.')
    parser.add_argument('--connectivity', default='TPE', type=str, help='Type of connectivity used for network '
                                                                        'construction. options: correlation, '
                                                                        'TE(tangent embedding), TPE(tangent pearson '
                                                                        'embedding), default: TPE.')
    parser.add_argument('--leave_one_out', default=False, type=str2bool, help='leave one site out CV instead of 10CV. '
                                                                              'Default: False.')
    parser.add_argument('--filename', default='tangent', type=str, help='filename for output file. default: tangent.')
    parser.add_argument('--ensemble', default=False, type=str2bool, help='Leave one site out, use ensemble MIDA/raw. '
                                                                         'Default: False')

    args = parser.parse_args()
    print('Arguments: \n', args)

    # Ridge classifier parameters
    params = dict()
    params['model'] = args.model  # MIDA, SMIDA or raw
    params['phenotypes'] = args.phenotypes  # Add phenotype features 
    params['seed'] = args.seed  # seed for random initialisation
    params['ensemble'] = args.ensemble

    # Algorithm choice
    params['algorithm'] = args.algorithm
    # Compute kernel statistical test of independence between features and site (boolean)
    params['KHSIC'] = args.KHSIC
    params['filename'] = args.filename  # Results output file    
    params['connectivity'] = args.connectivity  # Type of connectivity used for network construction
    params['atlas'] = args.atlas  # Atlas for network construction
    atlas = args.atlas  # Atlas for network construction (node definition)
    connectivity = args.connectivity  # Type of connectivity used for network construction

    # 10 Fold CV or leave one site out CV
    params['leave_one_out'] = args.leave_one_out
    if params['leave_one_out']:
        params['validation_ext'] = 'LOCV'
    else:
        params['validation_ext'] = '10CV'

    # Get subject IDs and class labels
    subject_ids = reader.get_ids()
    labels = reader.get_subject_score(subject_ids, score='DX_GROUP')

    # Number of subjects and classes for binary classification
    num_classes = 2
    num_subjects = len(subject_ids)
    params['n_subjects'] = num_subjects

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([num_subjects, num_classes])
    y = np.zeros([num_subjects, 1])

    # Get class labels for all subjects
    for i in range(num_subjects):
        y_data[i, int(labels[subject_ids[i]]) - 1] = 1
        y[i] = int(labels[subject_ids[i]])

    # Compute feature vectors (vectorised connectivity networks)
    if connectivity not in ['TE', 'TPE']:
        features = reader.get_networks(subject_ids, iter_no='', kind=connectivity, atlas_name=atlas)
    else:
        features = None

    # Source phenotype information and preprocess phenotypes

    if params['model'] == 'MIDA':
        pheno_ft = reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID', 'HANDEDNESS_CATEGORY', 'AGE_AT_SCAN',
                                                             'FIQ', 'VIQ', 'PIQ'], subject_ids)
    else:
        pheno_ft = reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID', 'EYE_STATUS_AT_SCAN',
                                                             'HANDEDNESS_CATEGORY', 'AGE_AT_SCAN', 'FIQ', 'VIQ', 'PIQ'],
                                                            subject_ids)

    pheno_ft.index = subject_ids
    pheno_ft2 = pheno_ft

    # number of sites available in the dataset
    params['num_domains'] = len(pheno_ft2['SITE_ID'].unique())

    # preprocess categorical data ordinally
    pheno_ft = reader.preprocess_phenotypes(pheno_ft, params)

    # construct phenotype feature vectors
    phenotype_ft = reader.phenotype_ft_vector(pheno_ft, num_subjects, params)

    if params['leave_one_out']:
        # leave one site out evaluation
        if params['ensemble']:
            train.leave_one_site_out_ensemble(params, num_subjects, subject_ids, features, y_data, y, phenotype_ft,
                                              pheno_ft)
        else:
            train.leave_one_site_out(params, num_subjects, subject_ids, features, y_data, y, phenotype_ft, pheno_ft)
    else:
        # 10 fold CV evaluation
        train.train_10CV(params, num_subjects, subject_ids, features, y_data, y, phenotype_ft, pheno_ft)


if __name__ == '__main__':
    main()
