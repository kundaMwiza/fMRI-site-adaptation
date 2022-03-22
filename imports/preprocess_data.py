# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
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

import csv
import glob
import os
import warnings

import numpy as np
import scipy.io as sio
# import re
# import sys
from nilearn import connectome
import pandas as pd
# from scipy.spatial import distance
# from scipy import signal
from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from .utils import root_dir_default, data_folder_name_default

warnings.filterwarnings("ignore")


def fetch_filenames(subject_ids, file_type, atlas, data_folder):
    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types
        filemapping  : resulting file name format

    returns:

        filenames    : list of filetypes (same length as subject_list)
    """

    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_' + atlas: '_rois_' + atlas + '.1D'}
    # The list to be filled
    filenames = []
    # data_folder = data_path.get_data_folder()
    # Fill list with requested file paths
    for i in range(len(subject_ids)):
        os.chdir(data_folder)
        try:
            try:
                os.chdir(data_folder)
                filenames.append(glob.glob('*' + subject_ids[i] + filemapping[file_type])[0])
            except:
                os.chdir(data_folder + '/' + subject_ids[i])
                filenames.append(glob.glob('*' + subject_ids[i] + filemapping[file_type])[0])
        except IndexError:
            filenames.append('N/A')
    return filenames


# Get timeseries arrays for list of subjects
def get_timeseries(subject_list, atlas_name, data_path, silence=False):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200

    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

    timeseries = []
    for i in range(len(subject_list)):
        subject_folder = os.path.join(data_path, subject_list[i])
        ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
        fl = os.path.join(subject_folder, ro_file[0])
        if not silence:
            print("Reading timeseries file %s" % fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries


# def get_conn_vec(data, connectivity, discard_diagonal=False):
#     setattr(connectivity, "vectorize", True)
#     setattr(connectivity, "discard_diagonal", discard_diagonal)
#     return connectivity.transform(data)


def vec2mat(conn_vec, n_rois, discard_diagonal=False):
    """
    nilearn takes the lower triangle of connectivity matrix
    """
    conn_mat = np.ones((n_rois, n_rois))
    if discard_diagonal:
        il = np.tril_indices(n_rois, k=-1)
    else:
        il = np.tril_indices(n_rois, k=0)
    conn_mat[il] = conn_vec
    conn_mat = conn_mat.T
    conn_mat[il] = conn_vec

    return conn_mat


#  compute connectivity matrices
def subject_connectivity(
        timeseries,
        # subjects,
        atlas,
        kind,
        # iter_no='',
        # seed=1234,
        # validation_ext='10CV',
        # n_subjects='',
        save=True,
        out_path=None):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subjects     : subject IDs
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        iter_no      : tangent connectivity iteration number for cross validation evaluation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    if kind in ['tangent', 'correlation', 'partial correlation', 'covariance']:
        input_data = timeseries.copy()
        conn_measure = connectome.ConnectivityMeasure(kind=kind, vectorize=True, discard_diagonal=True)
        if kind == 'tangent':
            # discard_diagonal = True
            setattr(conn_measure, "discard_diagonal", False)
    elif kind == 'TPE':
        conn_measure = connectome.ConnectivityMeasure(kind='correlation')
        input_data = conn_measure.fit_transform(timeseries)
        conn_measure = connectome.ConnectivityMeasure(kind='tangent', vectorize=True, discard_diagonal=False)
    else:
        raise ValueError("Unsupported connectivity %s" % kind)

    conn_measure.fit(input_data)
    # connectivity = conn_measure.transform(data)
    # conn_vec = get_conn_vec(data, connectivity_fit, discard_diagonal)
    conn_vec = conn_measure.transform(input_data)

    if save:
        if out_path is None:
            out_path = os.path.join(root_dir_default, data_folder_name_default)
        out_vec_file = os.path.join(out_path, "%s_%s.mat" % (atlas, kind))
        sio.savemat(out_vec_file, {'connectivity': conn_vec})
        # if kind != "TPE":
        #     for i, subj_id in enumerate(subjects):
        #         subject_file = os.path.join(save_path, subj_id,
        #                                     subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        #         sio.savemat(subject_file, {'connectivity': connectivity[i]})
        #     return connectivity
        # else:
        #     for i, subj_id in enumerate(subjects):
        #         subject_file = os.path.join(save_path, subj_id,
        #                                     subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '_' + str(
        #                                         iter_no) + '_' + str(seed) + '_' + validation_ext + str(
        #                                         n_subjects) + '.mat')
        #         sio.savemat(subject_file, {'connectivity': connectivity[i]})
        #     return connectivity_fit


# Get the list of subject IDs
def get_ids(fpath, num_subjects=None):
    """

    return:
        subject_ids    : list of all subject IDs
    """

    subject_ids = np.genfromtxt(os.path.join(fpath, 'subject_ids.txt'), dtype=str)

    if num_subjects is not None:
        subject_ids = subject_ids[:num_subjects]

    return subject_ids


# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score, pheno_fpath):
    scores_dict = {}
    with open(pheno_fpath) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                if score == 'HANDEDNESS_CATEGORY':
                    if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                        scores_dict[row['SUB_ID']] = 'R'
                    elif row[score] == 'Mixed':
                        scores_dict[row['SUB_ID']] = 'Ambi'
                    elif row[score] == 'L->R':
                        scores_dict[row['SUB_ID']] = 'Ambi'
                    else:
                        scores_dict[row['SUB_ID']] = row[score]
                elif score == 'FIQ' or score == 'PIQ' or score == 'VIQ':
                    if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                        scores_dict[row['SUB_ID']] = 100
                    else:
                        scores_dict[row['SUB_ID']] = float(row[score])

                else:
                    scores_dict[row['SUB_ID']] = row[score]

    return scores_dict


# preprocess phenotypes. Categorical -> ordinal representation
def preprocess_phenotypes(pheno_ft, params):
    if params['model'] == 'MIDA':
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2])], remainder='passthrough')
    else:
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2, 3])], remainder='passthrough')

    pheno_ft = ct.fit_transform(pheno_ft)
    pheno_ft = pheno_ft.astype('float32')
    return pheno_ft


# create phenotype feature vector to concatenate with fmri feature vectors
def phenotype_ft_vector(pheno_ft, num_subjects, params):
    gender = pheno_ft[:, 0]
    if params['model'] == 'MIDA':
        eye = pheno_ft[:, 0]
        hand = pheno_ft[:, 2]
        age = pheno_ft[:, 3]
        fiq = pheno_ft[:, 4]
    else:
        eye = pheno_ft[:, 2]
        hand = pheno_ft[:, 3]
        age = pheno_ft[:, 4]
        fiq = pheno_ft[:, 5]

    phenotype_ft = np.zeros((num_subjects, 4))
    phenotype_ft_eye = np.zeros((num_subjects, 2))
    phenotype_ft_hand = np.zeros((num_subjects, 3))

    for i in range(num_subjects):
        phenotype_ft[i, int(gender[i])] = 1
        phenotype_ft[i, -2] = age[i]
        phenotype_ft[i, -1] = fiq[i]
        phenotype_ft_eye[i, int(eye[i])] = 1
        phenotype_ft_hand[i, int(hand[i])] = 1

    if params['model'] == 'MIDA':
        phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand], axis=1)
    else:
        phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand, phenotype_ft_eye], axis=1)

    return phenotype_ft


# Load precomputed fMRI connectivity networks
def get_networks(subject_list, kind, data_path, iter_no='', seed=1234, validation_ext='10CV', n_subjects='',
                 atlas="aal", variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        data_path    : DataPath.
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks

    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    # all_networks = []
    # for subject in subject_list:
    #     if len(kind.split()) == 2:
    #         kind = '_'.join(kind.split())
    #     if kind not in ['TPE', 'tangent']:
    #         fl = os.path.join(data_folder, subject,
    #                           subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")
    #     else:
    #         fl = os.path.join(data_folder, subject,
    #                           subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + '_' + str(
    #                               iter_no) + '_' + str(seed) + '_' + validation_ext + str(n_subjects) + ".mat")
    #
    #     matrix = sio.loadmat(fl)[variable]
    #     all_networks.append(matrix)
    #
    # if kind in ['tangent', 'TPE']:
    #     norm_networks = [mat for mat in all_networks]
    # else:
    #     norm_networks = [np.arctanh(mat) for mat in all_networks]
    #
    # idx = np.triu_indices_from(all_networks[0], 1)
    # vec_networks = [mat[idx] for mat in norm_networks]
    # matrix = np.vstack(vec_networks)
    fl = os.path.join(data_path, "%s_%s.mat" % (atlas, kind))
    conn_matrix = sio.loadmat(fl)[variable]

    return conn_matrix


# Construct the adjacency matrix of the population from phenotypic scores
def create_affinity_graph_from_scores(scores, subject_list, pheno_fpath):
    """
        scores       : list of phenotypic information to be used to construct the affinity graph
        subject_list : list of subject IDs
    return:
        graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
    """

    num_nodes = len(subject_list)
    pheno_ft = pd.DataFrame()
    global_phenos = []

    for i, l in enumerate(scores):
        phenos = []
        label_dict = get_subject_score(subject_list, l, pheno_fpath)

        # quantitative phenotypic scores
        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                phenos.append(float(label_dict[subject_list[k]]))
        else:
            for k in range(num_nodes):
                phenos.append(label_dict[subject_list[k]])
        global_phenos.append(phenos)

    for i, l in enumerate(scores):
        pheno_ft.insert(i, l, global_phenos[i], True)

    return pheno_ft
