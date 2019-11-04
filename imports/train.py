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


import os
import time
import pandas as pd
import numpy as np
from nilearn import connectome
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import sklearn.metrics
import scipy.stats as sc
import scipy.io as sio
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from imports import KHSIC as KHSIC
from imports import MIDA as MIDA
from imports import preprocess_data as Reader

root_folder = '/path/to/data/'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal/')

# Transform test data using the transformer learned on the training data
def process_test_data(timeseries, transformer, ids, params, k, seed, validation_ext):
    conn_measure = connectome.ConnectivityMeasure(kind='correlation')
    test_data = conn_measure.fit_transform(timeseries)

    if params['connectivity'] == 'TE':
        connectivity = transformer.transform(timeseries)
    else:
        connectivity = transformer.transform(test_data)

    save_path = data_folder
    atlas_name = params['atlas']
    kind = params['connectivity']

    for i, subj_id in enumerate(ids):
        subject_file = os.path.join(save_path, subj_id,
                        subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '_' + str(k) + '_' + str(seed) + '_' + validation_ext + str(params['n_subjects'])+'.mat')
        sio.savemat(subject_file, {'connectivity': connectivity[i]})  
    
# Process timeseries for tangent train/test split
def process_timeseries(subject_IDs, train_ind, test_ind, params, k, seed, validation_ext):
    atlas = params['atlas']
    kind = params['connectivity']
    timeseries = Reader.get_timeseries(subject_IDs, atlas, silence=True)
    train_timeseries = [timeseries[i] for i in train_ind]
    subject_IDs_train = [subject_IDs[i] for i in train_ind]
    test_timeseries = [timeseries[i] for i in test_ind]
    subject_IDs_test = [subject_IDs[i] for i in test_ind]
    
    print('computing tangent connectivity features..')
    transformer = Reader.subject_connectivity(train_timeseries, subject_IDs_train, atlas, kind, k, seed, validation_ext, n_subjects=params['n_subjects'])
    test_data_save = process_test_data(test_timeseries, transformer, subject_IDs_test, params, k, seed, validation_ext)   
    
# Grid search CV 
def grid_search(params, train_ind, test_ind, features, y, y_data, phenotype_ft=None, domain_ft=None, label_info=None):
    
    # MIDA parameter search
    mu_vals = [0.5, 0.75, 1.0]
    h_vals = [50, 150, 300]

    # Add phenotypes or not
    add_phenotypes = params['phenotypes']
    
    # Algorithm choice
    algorithm = params['algorithm']

    # Model choice
    model = params['model']

    # Seed
    seed = params['seed']

    # best parameters and 5CV accuracy
    best_model = {}
    best_model['acc'] = 0

    # Grid search formulation
    if algorithm in ['LR', 'SVM']:
        C_vals = [1, 5, 10]
        if algorithm == 'LR':
            max_iter_vals = [100000]
            alg = LogisticRegression(random_state=seed, solver='lbfgs')
        else:
            max_iter_vals = [100000]
            alg = svm.SVC(random_state=seed, kernel='linear')
        parameters = {'C': C_vals, 'max_iter': max_iter_vals}
    else:
        alpha_vals = [0.25, 0.5, 0.75]
        parameters = {'alpha': alpha_vals}
        alg = RidgeClassifier(random_state=seed)
    
    if model in ['MIDA', 'SMIDA']:
        for mu in mu_vals:
            for h in h_vals:
                x_data = features
                x_data = MIDA.MIDA(x_data, domain_ft, mu=mu, h=h, labels=False)
                if add_phenotypes == True:
                    x_data = np.concatenate([x_data, phenotype_ft], axis=1)
                clf = GridSearchCV(alg, parameters, cv=5)
                clf.fit(x_data[train_ind], y[train_ind].ravel())
                if clf.best_score_ > best_model['acc']:
                    best_model['mu'] = mu
                    best_model['h'] = h
                    best_model = dict(best_model, **clf.best_params_)
                    best_model['acc'] = clf.best_score_

    else:
        x_data = features
        if add_phenotypes == True:
            x_data = np.concatenate([x_data, phenotype_ft], axis=1)
        clf = GridSearchCV(alg, parameters, cv=5)
        clf.fit(x_data[train_ind], y[train_ind].ravel())
        if clf.best_score_ > best_model['acc']:
            best_model = dict(best_model, **clf.best_params_)
            best_model['acc'] = clf.best_score_ 
    
    return best_model

# Ensemble models with different FC measures 
def leave_one_site_out_ensemble(params, num_subjects, subject_IDs, features, y_data, y, phenotype_ft, phenotype_raw):
    results_acc = []
    results_auc = []
    all_pred_acc = np.zeros(y.shape)
    all_pred_auc = np.zeros(y.shape)

    algorithm = params['algorithm']
    seed = params['seed']
    atlas = params['atlas']
    num_domains = params['num_domains']
    validation_ext = params['validation_ext']
    filename = params['filename']
    connectivities = {0: 'correlation', 1: 'TPE', 2: 'TE'}
    features_c = Reader.get_networks(subject_IDs, iter_no='', kind='correlation', atlas_name=atlas)
    add_phenotypes = params['phenotypes']


    for i in range(num_domains):
        k = i
        train_ind = np.where(phenotype_raw[:, 1] != i)[0]
        test_ind = np.where(phenotype_raw[:, 1] == i)[0]

        # load tangent pearson features
        try:
            features_t = Reader.get_networks(subject_IDs, iter_no=k, seed=seed, validation_ext=validation_ext, kind='TPE', n_subjects=params['n_subjects'], atlas_name=atlas)
        except:
            print("Tangent features not found. reloading timeseries data")
            time.sleep(10)
            params['connectivity'] = 'TPE'
            process_timeseries(subject_IDs, train_ind, test_ind, params, k, seed, validation_ext)
            features_t = Reader.get_networks(subject_IDs, iter_no=k, seed=seed, validation_ext=validation_ext, kind='TPE', n_subjects=params['n_subjects'], atlas_name=atlas)
        
        # load tangent timeseries features
        try:
            features_tt = Reader.get_networks(subject_IDs, iter_no=k, seed=seed, validation_ext=validation_ext, kind='TE', n_subjects=params['n_subjects'], atlas_name=atlas)
        except:
            print("Tangent features not found. reloading timeseries data")
            time.sleep(10)
            params['connectivity'] = 'TE'
            process_timeseries(subject_IDs, train_ind, test_ind, params, k, seed, validation_ext)
            features_tt = Reader.get_networks(subject_IDs, iter_no=k, seed=seed, validation_ext=validation_ext, kind='TE', n_subjects=params['n_subjects'], atlas_name=atlas)
        
        # all loaded features
        features = [features_c, features_t, features_tt]
        
        all_best_models = []
        x_data_ft = []
        if params['model'] == 'MIDA':
            domain_ft = MIDA.site_information_mat(phenotype_raw, num_subjects, num_domains)
            for ft in range(3):
                best_model = grid_search(params, train_ind, test_ind, features[ft], y, y_data, phenotype_ft=phenotype_ft, domain_ft=domain_ft)
                print('for', connectivities[ft], ', best parameters from 5CV grid search are: \n', best_model)
                x_data = MIDA.MIDA(features[ft], domain_ft, mu=best_model['mu'], h=best_model['h'], labels=False)
                best_model.pop('mu')
                best_model.pop('h')
                best_model.pop('acc')
                all_best_models.append(best_model)
                x_data_ft.append(x_data)

        else:
            for ft in range(3):
                best_model = grid_search(params, train_ind, test_ind, features[ft], y, y_data)
                print('best parameters from 5CV grid search are: \n', best_model)
                best_model.pop('acc')
                all_best_models.append(best_model)
                x_data_ft.append(features[ft])

        
        algs = []
        preds_binary = []
        preds_decision = []

        # fit and compute predictions from all three models
        for ft in range(3):
            if add_phenotypes == True:
                x_data = np.concatenate([x_data, phenotype_ft], axis=1)

            if algorithm == 'LR':
                clf = LogisticRegression(random_state=seed, solver='lbfgs', **all_best_models[ft])
            elif algorithm == 'SVM':
                clf = svm.SVC(kernel='linear', random_state=seed, **all_best_models[ft])
            else:
                clf = RidgeClassifier(random_state=seed, **all_best_models[ft])
            
            algs.append(clf.fit(x_data_ft[ft][train_ind], y[train_ind].ravel()))
            preds_binary.append(clf.predict(x_data_ft[ft][test_ind]))
            preds_decision.append(clf.decision_function(x_data_ft[ft][test_ind]))
        
        # mode prediciton
        mode_predictions = sc.mode(np.hstack([preds_binary[j][np.newaxis].T for j in range(3)]), axis = 1)[0].ravel()
        all_pred_acc[test_ind, :] = mode_predictions[:, np.newaxis]
        
        # Compute the accuracy
        lin_acc = accuracy_score(y[test_ind].ravel(), mode_predictions)

        # mean decision score
        mean_predictions = np.hstack([preds_decision[j][:, np.newaxis] for j in range(3)]).mean(axis=1)
        all_pred_auc[test_ind, :] = mean_predictions[:, np.newaxis]

        # Compute the AUC
        lin_auc = sklearn.metrics.roc_auc_score(y[test_ind], mean_predictions)
        
        # append accuracy and AUC to respective lists
        results_acc.append(lin_acc)
        results_auc.append(lin_auc)
        print("-"*100)
        print("Fold number: %d" % k)
        print("Linear Accuracy: " + str(lin_acc))
        print("Linear AUC: "+str(lin_auc))
        print("-"*100)

    avg_acc = np.array(results_acc).mean()  
    std_acc = np.array(results_acc).std()
    avg_auc = np.array(results_auc).mean()
    std_auc = np.array(results_auc).std()
    weighted_acc = (y == all_pred_acc).sum()/params['n_subjects']
    weighted_auc = sklearn.metrics.roc_auc_score(y, all_pred_auc)

    print("accuracy average", avg_acc)
    print("standard deviation accuracy", std_acc)
    print("auc average", avg_auc)
    print("standard deviation auc", std_auc)
    print("(weighted) accuracy",  weighted_acc)
    print("(weighted) auc",  weighted_auc)

    all_results = pd.DataFrame()
    all_results['ACC'] = results_acc
    all_results['AUC'] = results_auc
    all_results.to_csv(filename + '.csv')

# leave one site out application performance
def leave_one_site_out(params, num_subjects, subject_IDs, features, y_data, y, phenotype_ft, phenotype_raw):

    results_acc = []
    results_auc = []
    all_pred_acc = np.zeros(y.shape)
    all_pred_auc = np.zeros(y.shape)

    algorithm = params['algorithm']
    seed = params['seed']
    connectivity = params['connectivity']
    atlas = params['atlas']
    num_domains = params['num_domains']
    validation_ext = params['validation_ext']
    filename = params['filename']
    add_phenotypes = params['phenotypes']

    for i in range(num_domains):
        k = i
        train_ind = np.where(phenotype_raw[:, 1] != i)[0]
        test_ind = np.where(phenotype_raw[:, 1] == i)[0]
            
        if connectivity in ['TPE', 'TE']:
            try:
                features = Reader.get_networks(subject_IDs, iter_no=k, seed=seed, validation_ext=validation_ext, kind=connectivity, n_subjects=params['n_subjects'], atlas_name=atlas)
            except:
                print("Tangent features not found. reloading timeseries data")
                time.sleep(10)
                process_timeseries(subject_IDs, train_ind, test_ind, params, k, seed, validation_ext)
                features = Reader.get_networks(subject_IDs, iter_no=k, seed=seed, validation_ext=validation_ext, kind=connectivity, n_subjects=params['n_subjects'], atlas_name=atlas)

        if params['model'] == 'MIDA':
            domain_ft = MIDA.site_information_mat(phenotype_raw, num_subjects, num_domains)
            best_model = grid_search(params, train_ind, test_ind, features, y, y_data, phenotype_ft=phenotype_ft, domain_ft=domain_ft)
            print('best parameters from 5CV grid search: \n', best_model)
            x_data = MIDA.MIDA(features, domain_ft, mu=best_model['mu'], h=best_model['h'], labels=False)
            best_model.pop('mu')
            best_model.pop('h')
        else:
            best_model = grid_search(params, train_ind, test_ind, features, y, y_data, phenotype_ft=phenotype_ft)
            print('best parameters from 5CV grid search: \n', best_model)
            x_data = features
        
        if add_phenotypes == True:
            x_data = np.concatenate([x_data, phenotype_ft], axis=1)

        # Remove accuracy key from best model dictionary
        best_model.pop('acc')
        
        # Set classifier
        if algorithm == 'LR':
            clf = LogisticRegression(random_state=seed, solver='lbfgs', **best_model)
        elif algorithm == 'SVM':
            clf = svm.SVC(random_state=seed, kernel='linear', **best_model)
        else:
            clf = RidgeClassifier(random_state=seed, **best_model)

        # Fit classifier
        clf.fit(x_data[train_ind, :], y[train_ind].ravel())

        # Compute the accuracy
        lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
        y_pred = clf.predict(x_data[test_ind, :])
        all_pred_acc[test_ind, :] = y_pred[:, np.newaxis]

        # Compute the AUC
        pred = clf.decision_function(x_data[test_ind, :])
        all_pred_auc[test_ind, :] = pred[:, np.newaxis]
        lin_auc = sklearn.metrics.roc_auc_score(y[test_ind], pred)
        
        # append accuracy and AUC to respective lists
        results_acc.append(lin_acc)
        results_auc.append(lin_auc)
        print("-"*100)
        print("Fold number: %d" % k)
        print("Linear Accuracy: " + str(lin_acc))
        print("Linear AUC: "+str(lin_auc))
        print("-"*100)

    avg_acc = np.array(results_acc).mean()  
    std_acc = np.array(results_acc).std()
    avg_auc = np.array(results_auc).mean()
    std_auc = np.array(results_auc).std()
    weighted_acc = (y == all_pred_acc).sum()/params['n_subjects']
    weighted_auc = sklearn.metrics.roc_auc_score(y, all_pred_auc)
    
    print("(unweighted) accuracy average", avg_acc)
    print("(unweighted) standard deviation accuracy", std_acc)
    print("(unweighted) auc average", avg_auc)
    print("(unweighted) standard deviation auc", std_auc)
    print("(weighted) accuracy",  weighted_acc)
    print("(weighted) auc",  weighted_auc)

    all_results = pd.DataFrame()
    all_results['ACC'] = results_acc
    all_results['AUC'] = results_auc
    all_results.to_csv(filename + '.csv')


# 10 fold CV 
def train_10CV(params, num_subjects, subject_IDs, features, y_data, y, phenotype_ft, phenotype_raw):
    
    results_acc = []
    results_auc = []

    algorithm = params['algorithm']
    seed = params['seed']
    connectivity = params['connectivity']
    atlas = params['atlas']
    num_domains = params['num_domains']
    model = params['model']
    add_phenotypes = params['phenotypes']
    filename = params['filename']
    validation_ext = params['validation_ext']

    if seed == 123:
        skf = StratifiedKFold(n_splits=10)
    else:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    for sets, k in zip(list(reversed(list(skf.split(np.zeros(num_subjects), np.squeeze(y))))), list(range(10))):            
        train_ind = sets[0]
        test_ind = sets[1]

        if connectivity in ['TPE', 'TE']:
            try:
                features = Reader.get_networks(subject_IDs, iter_no=k, seed=seed, validation_ext=validation_ext, kind=connectivity, n_subjects=params['n_subjects'], atlas_name=atlas)
            except:
                print("Tangent features not found. reloading timeseries data")
                time.sleep(10)
                process_timeseries(subject_IDs, train_ind, test_ind, params, k, seed, validation_ext)
                features = Reader.get_networks(subject_IDs, iter_no=k, seed=seed, validation_ext=validation_ext, kind=connectivity, n_subjects=params['n_subjects'], atlas_name=atlas)

        
        if model == 'MIDA':
            domain_ft = MIDA.site_information_mat(phenotype_raw, num_subjects, num_domains)
            best_model = grid_search(params, train_ind, test_ind, features, y, y_data, phenotype_ft=phenotype_ft, domain_ft=domain_ft)
            print('best parameters from 5CV grid search: \n', best_model)
            x_data = MIDA.MIDA(features, domain_ft, mu=best_model['mu'], h=best_model['h'], labels=False)
            best_model.pop('mu')
            best_model.pop('h')

        else:
            best_model = grid_search(params, train_ind, test_ind, features, y, y_data, phenotype_ft=phenotype_ft)
            print('best parameters from 5CV grid search: \n', best_model)
            x_data = features

        if add_phenotypes == True:
            x_data = np.concatenate([x_data, phenotype_ft], axis=1)
        
        # Remove accuracy key from best model dictionary
        best_model.pop('acc')
        
        # Set classifier
        if algorithm == 'LR':
            clf = LogisticRegression(random_state=seed, solver='lbfgs', **best_model)

        elif algorithm == 'SVM':
            clf = svm.SVC(random_state=seed, kernel='linear', **best_model)
        else:
            clf = RidgeClassifier(random_state=seed, **best_model)

        clf.fit(x_data[train_ind, :], y[train_ind].ravel())

        # Compute the accuracy
        lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())

        # Compute the AUC
        pred = clf.decision_function(x_data[test_ind, :])
        lin_auc = sklearn.metrics.roc_auc_score(y[test_ind], pred)
        
        # append accuracy and AUC to respective lists
        results_acc.append(lin_acc)
        results_auc.append(lin_auc)
        print("-"*100)
        print("Fold number: %d" % k)
        print("Linear Accuracy: " + str(lin_acc))
        print("Linear AUC: "+str(lin_auc))
        print("-"*100)

    avg_acc = np.array(results_acc).mean()  
    std_acc = np.array(results_acc).std()
    avg_auc = np.array(results_auc).mean()
    std_auc = np.array(results_auc).std()
    print("accuracy average", avg_acc)
    print("standard deviation accuracy", std_acc)
    print("auc average", avg_auc)
    print("standard deviation auc", std_auc)

    # compute statistical test of independence
    if params['KHSIC'] == True and model == 'MIDA':
        test_stat, threshold, pval = KHSIC.hsic_gam(features, domain_ft, alph = 0.05)
        pval = 1-pval
        print('KHSIC sample value: %.2f' % test_stat,'Threshold: %.2f' % threshold, 'p value: %.10f' % pval) 

    all_results = pd.DataFrame()
    all_results['ACC'] = results_acc
    all_results['AUC'] = results_auc
    all_results.to_csv(filename + '.csv')
