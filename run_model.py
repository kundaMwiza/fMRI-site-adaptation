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
from imports import preprocess_data as Reader
from imports import train as train
import warnings
warnings.filterwarnings("ignore")

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

def main():

    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a Ridge classifier. '
                                                'MIDA is used to minimize the distribution mismatch between ABIDE sites')
    parser.add_argument('--atlas', default='cc200', help='Atlas for network construction (node definition) options: ho, cc200, cc400, default: cc200.')
    parser.add_argument('--model', default='MIDA', type=str, help='Options: MIDA, raw. default: MIDA.')
    parser.add_argument('--algorithm', default='Ridge', type=str, help='Options: Ridge, LR (Logistic regression),' 
                                                                        ' SVM (Support vector machine). default: Ridge.')
    parser.add_argument('--phenotypes', default=True, type=str2bool, help='Add phenotype features. default: True.')
    parser.add_argument('--KHSIC', default=True, type=str2bool, help='Compute kernel statistical test of independence between features'
                                                        ' and site, default True.')
    parser.add_argument('--seed', default=1234, type=int, help='Seed for random initialisation. default: 1234.')
    parser.add_argument('--connectivity', default='TPE', type=str, help='Type of connectivity used for network '
                                                                    'construction. options: correlation, TE(tangent embedding), TPE(tangent pearson embedding),'
                                                                    'default: TPE.')
    parser.add_argument('--leave_one_out', default=False, type=str2bool, help='leave one site out CV instead of 10CV. default: False.')
    parser.add_argument('--filename', default='tangent', type=str, help='filename for output file. default: tangent.')
    parser.add_argument('--ensemble', default=False, type=str2bool, help='Leave one site out, use ensemble MIDA/raw. default: False')
    
    args = parser.parse_args()  
    print('Arguments: \n', args)

    # Ridge classifier parameters
    params = dict()

    params['model'] = args.model                    # MIDA, SMIDA or raw
    params['phenotypes'] = args.phenotypes          # Add phenotype features 
    params['seed'] = args.seed                      # seed for random initialisation
    params['ensemble'] = args.ensemble
    
    # Algorithm choice
    params['algorithm'] = args.algorithm    

    params['KHSIC'] = args.KHSIC                    # Compute kernel statistical test of independence between features and site (boolean)

    params['filename'] = args.filename              # Results output file    
    params['connectivity'] = args.connectivity      # Type of connectivity used for network construction
    params['atlas'] = args.atlas                    # Atlas for network construction
    atlas = args.atlas                              # Atlas for network construction (node definition)
    connectivity = args.connectivity                # Type of connectivity used for network construction


    # 10 Fold CV or leave one site out CV
    params['leave_one_out'] = args.leave_one_out
    if params['leave_one_out'] == True:
        params['validation_ext'] = 'LOCV'
    else:
        params['validation_ext'] = '10CV'
    
    # Get subject IDs and class labels
    subject_IDs = Reader.get_ids()
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
    
    # Number of subjects and classes for binary classification
    num_classes = 2
    num_subjects = len(subject_IDs)
    params['n_subjects'] = num_subjects

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([num_subjects, num_classes])
    y = np.zeros([num_subjects, 1])

    # Get class labels for all subjects
    for i in range(num_subjects):
        y_data[i, int(labels[subject_IDs[i]])-1] = 1
        y[i] = int(labels[subject_IDs[i]])

    # Compute feature vectors (vectorised connectivity networks)
    if connectivity not in ['TE', 'TPE']:
        features = Reader.get_networks(subject_IDs, iter_no='', kind=connectivity, atlas_name=atlas)
    else:
        features = None

    # Source phenotype information and preprocess phenotypes
    
    if params['model'] == 'MIDA':
        pheno_ft = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID', 'HANDEDNESS_CATEGORY', 'AGE_AT_SCAN','FIQ', 'VIQ', 'PIQ'], subject_IDs)
    else:
        pheno_ft = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID','EYE_STATUS_AT_SCAN', 'HANDEDNESS_CATEGORY', 'AGE_AT_SCAN','FIQ', 'VIQ', 'PIQ'], subject_IDs)

    pheno_ft.index = subject_IDs
    pheno_ft2 = pheno_ft

    # number of sites available in the dataset
    params['num_domains'] = len(pheno_ft2['SITE_ID'].unique())

    # preprocess categorical data ordinally
    pheno_ft = Reader.preprocess_phenotypes(pheno_ft, params)
    
    # construct phenotype feature vectors
    phenotype_ft = Reader.phenotype_ft_vector(pheno_ft, num_subjects, params)

    if params['leave_one_out'] == True:
        # leave one site out evaluation
        if params['ensemble'] == True:
            train.leave_one_site_out_ensemble(params, num_subjects, subject_IDs, features, y_data, y, phenotype_ft, pheno_ft)
        else:
            train.leave_one_site_out(params, num_subjects, subject_IDs, features, y_data, y, phenotype_ft, pheno_ft)
    else:
        # 10 fold CV evaluation
        train.train_10CV(params, num_subjects, subject_IDs, features, y_data, y, phenotype_ft, pheno_ft)


if __name__ == '__main__':
    main()



    


