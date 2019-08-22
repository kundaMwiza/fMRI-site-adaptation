import time
import warnings
import os
import argparse
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import distance
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import sklearn.metrics
import scipy.io as sio
import preprocess_data as Reader
import KHSIC as KHSIC
import MIDA as MIDA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from nilearn import connectome
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import warnings


# root_folder = r'path/to/data'
root_folder = '/Users/mrwiz/Downloads/population-gcn-master/gcn/Data/'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal/')

# Transform test data using the transformer learned on the training data
def process_test_data(timeseries, transformer, ids, params, k, seed, validation_ext):
    conn_measure = connectome.ConnectivityMeasure(kind='correlation')
    test_data = conn_measure.fit_transform(timeseries)

    if params['connectivity'] == 'timetangent':
        connectivity = transformer.transform(timeseries)
    else:
        connectivity = transformer.transform(test_data)

    save_path = data_folder
    atlas_name = params['atlas']
    kind = params['connectivity']

    for i, subj_id in enumerate(ids):
        subject_file = os.path.join(save_path, subj_id,
                        subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '_' + str(k) + '_' + str(seed) + '_' + validation_ext + '.mat')
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
    transformer = Reader.subject_connectivity(train_timeseries, subject_IDs_train, atlas, kind, k, seed, validation_ext)
    test_data_save = process_test_data(test_timeseries, transformer, subject_IDs_test, params, k, seed, validation_ext)   
    
# Grid search CV 
def grid_search(params, train_ind, test_ind, features, y, phenotype_ft=None, domain_ft=None):
    
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
            alg = LinearSVC(random_state=seed)
        parameters = {'C': C_vals, 'max_iter': max_iter_vals}
    else:
        alpha_vals = [0.25, 0.5, 0.75]
        parameters = {'alpha': alpha_vals}
        alg = RidgeClassifier(random_state=seed)
    
    if model == 'MIDA':
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

# leave one site out application performance
def leave_one_site_out(params, num_subjects, subject_IDs, features, y_data, y, phenotype_ft, phenotype_raw):

    results_acc = []
    results_auc = []

    algorithm = params['algorithm']
    seed = params['seed']
    connectivity = params['connectivity']
    atlas = params['atlas']
    num_domains = params['num_domains']
    validation_ext = params['validation_ext']

    for i in range(num_domains):
        k = i
        train_ind = np.where(phenotype_raw[:, 1] != i)[0]
        test_ind = np.where(phenotype_raw[:, 1] == i)[0]
            
        if connectivity in ['tangent', 'timetangent']:
            try:
                features = Reader.get_networks(subject_IDs, iter_no=k, validation_ext=validation_ext, kind=connectivity, atlas_name=atlas)
            except FileNotFoundError:
                print("Tangent features not found. reloading timeseries data")
                time.sleep(10)
                process_timeseries(subject_IDs, train_ind, test_ind, params, k, seed, validation_ext)
                features = Reader.get_networks(subject_IDs, iter_no=k, validation_ext=validation_ext, kind=connectivity, atlas_name=atlas)

        if params['model'] == 'MIDA':
            domain_ft = MIDA.site_information_mat(phenotype_raw, num_subjects, num_domains)
            best_model = grid_search(params, train_ind, test_ind, features, y, domain_ft=domain_ft)
            print('best parameters from 5CV grid search: \n', best_model)
            x_data = MIDA.MIDA(features, domain_ft, mu=best_model['mu'], h=best_model['h'], labels=False)
            best_model.pop('mu')
            best_model.pop('h')
        else:
            print('best parameters from 5CV grid search: \n', best_model)
            best_model = grid_search(params, train_ind, test_ind, features, y)
            x_data = features
        
        # Remove accuracy key from best model dictionary
        best_model.pop('acc')
        
        # Set classifier
        if algorithm == 'LR':
            clf = LogisticRegression(random_state=seed, solver='lbfgs', **best_model)
        elif algorithm == 'SVM':
            clf = LinearSVC(random_state=seed, **best_model)
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

    all_results = pd.DataFrame()
    all_results['ACC'] = results_acc
    all_results['AUC'] = results_auc
    all_results.to_csv(filename + 'LOCV' + '.csv')

# 10 fold CV 
def train(params, num_subjects, subject_IDs, features, y_data, y, phenotype_ft, phenotype_raw):
    
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

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for sets, k in zip(list(reversed(list(skf.split(np.zeros(num_subjects), np.squeeze(y))))), list(range(10))):            
        train_ind = sets[0]
        test_ind = sets[1]

        if connectivity in ['tangent', 'timetangent']:
            try:
                features = Reader.get_networks(subject_IDs, iter_no=k, seed=seed, validation_ext=validation_ext, kind=connectivity, atlas_name=atlas) 
            except FileNotFoundError:
                print("Tangent features not found. reloading timeseries data")
                time.sleep(10)
                process_timeseries(subject_IDs, train_ind, test_ind, params, k, seed, validation_ext)
                features = Reader.get_networks(subject_IDs, iter_no=k, seed=seed, validation_ext=validation_ext, kind=connectivity, atlas_name=atlas) 

        
        if model == 'MIDA':
            domain_ft = MIDA.site_information_mat(phenotype_raw, num_subjects, num_domains)
            best_model = grid_search(params, train_ind, test_ind, features, y, phenotype_ft=phenotype_ft, domain_ft=domain_ft)
            print('best parameters from 5CV grid search: \n', best_model)
            x_data = MIDA.MIDA(features, domain_ft, mu=best_model['mu'], h=best_model['h'], labels=False)
            best_model.pop('mu')
            best_model.pop('h')

        else:
            best_model = grid_search(params, train_ind, test_ind, features, y, phenotype_ft=phenotype_ft)
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
            clf = LinearSVC(random_state=seed, **best_model)
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
    if params['KHSIC'] == True:
        test_stat, threshold = KHSIC.hsic_gam(features, domain_ft, alph = 0.05)
        print('KHSIC sample value: %.2f' % test_stat,'Threshold: %.2f' % threshold) 

    all_results = pd.DataFrame()
    all_results['ACC'] = results_acc
    all_results['AUC'] = results_auc
    all_results.to_csv(filename+'.csv')

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

    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a Ridge classifier'
                                                'MIDA is used to minimize the distribution mismatch between ABIDE sites')
    parser.add_argument('--atlas', default='cc200', help='Atlas for network construction (node definition) options: ho, cc200, cc400 (default: cc200), '
                                                    'see preprocessed-connectomes-project.org/abide/Pipelines.html '
                                                    'for more options )')
    parser.add_argument('--model', default='MIDA', type=str, help='Options: MIDA, raw. (default MIDA)')
    parser.add_argument('--algorithm', default='Ridge', type=str, help='Options: Ridge, LR (Logistic regression), SVM (Support vector machine). (default Ridge)')
    parser.add_argument('--phenotypes', default=True, type=str2bool, help='Add phenotype features (default True)')
    parser.add_argument('--KHSIC', default=False, type=str2bool, help='Compute kernel statistical test of independence between features'
                                                        'and site (default True)')
    parser.add_argument('--seed', default=1234, type=int, help='Seed for random initialisation (default: 1234)')
    parser.add_argument('--connectivity', default='tangent', type=str, help='Type of connectivity used for network '
                                                                    'construction (default: tangent, '
                                                                    'options: correlation, partial correlation, '
                                                                    'tangent)')
    parser.add_argument('--leave_one_out', default='False', type=str2bool, help='leave one site out CV instead of 10CV')
    parser.add_argument('--filename', default='tangent', type=str, help='results output file')

    args = parser.parse_args()  
    print('Arguments: \n', args)

    # Ridge classifier parameters
    params = dict()

    params['model'] = args.model                    # MIDA, SMIDA or raw
    params['phenotypes'] = args.phenotypes          # Add phenotype features 
    params['seed'] = args.seed                      # seed for random initialisation
    
    # Algorithm choice
    params['algorithm'] = args.algorithm    

    params['KHSIC'] = args.KHSIC                    # Compute kernel statistical test of independence between features and site (boolean)

    params['filename'] = args.filename              # Results output file    
    params['connectivity'] = args.connectivity      # Type of connectivity used for network construction
    params['atlas'] = args.atlas                    # Atlas for network construction
    atlas = args.atlas                              # Atlas for network construction (node definition)
    connectivity = args.connectivity                # Type of connectivity used for network construction

    if atlas == 'cc200':
        params['num_rois'] = 200
    elif atlas == 'ho':
        params['num_rois'] = 110
    elif atlas == 'cc400':
        params['num_rois'] = 392
    
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

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([num_subjects, num_classes])
    y = np.zeros([num_subjects, 1])

    # Get class labels for all subjects
    for i in range(num_subjects):
        y_data[i, int(labels[subject_IDs[i]])-1] = 1
        y[i] = int(labels[subject_IDs[i]])

    # Compute feature vectors (vectorised connectivity networks)
    if connectivity not in ['tangent', 'timetangent']:
        features = Reader.get_networks(subject_IDs, iter_no='', kind=connectivity, atlas_name=atlas)
    else:
        features = None

    # Source phenotype information and preprocess phenotypes
    pheno_ft = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID','EYE_STATUS_AT_SCAN', 'HANDEDNESS_CATEGORY', 'AGE_AT_SCAN','FIQ', 'VIQ', 'PIQ'], subject_IDs)
    pheno_ft.index = subject_IDs
    pheno_ft2 = pheno_ft

    # number of sites available in the dataset
    params['num_domains'] = len(pheno_ft2['SITE_ID'].unique())

    # preprocess categorical data ordinally
    pheno_ft = Reader.preprocess_phenotypes(pheno_ft)
    
    # construct phenotype feature vectors
    phenotype_ft = Reader.phenotype_ft_vector(pheno_ft, num_subjects)

    if params['leave_one_out'] == True:
        # leave one site out evaluation
        params['phenotypes'] = False
        leave_one_site_out(params, num_subjects, subject_IDs, features, y_data, y, phenotype_ft, pheno_ft)
    else:
        # 10 fold CV evaluation
        train(params, num_subjects, subject_IDs, features, y_data, y, phenotype_ft, pheno_ft)


if __name__ == '__main__':
    main()



    


