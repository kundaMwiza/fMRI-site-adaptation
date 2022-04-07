import scipy.io as sio
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from imports import preprocess_data as reader


root_dir = '/media/shuo/MyDrive/data/brain/'
site_label_path = "/media/shuo/MyDrive/data/brain/ABIDE_pcp/cpac/filt_noglobal/site_label.mat"
pheno_fpath = os.path.join(root_dir, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
data_folder = os.path.join(root_dir, 'ABIDE_pcp/cpac/filt_noglobal/')

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
betas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
dims = [5, 10, 20, 50, 100]

site_label = sio.loadmat(site_label_path)['site_label'][0]

subject_ids = reader.get_ids(data_folder)
labels = reader.get_subject_score(subject_ids, score='DX_GROUP', pheno_fpath=pheno_fpath)
y = np.zeros(len(subject_ids))
for i in range(len(subject_ids)):
    y[i] = int(labels[subject_ids[i]])
X = reader.get_networks(subject_ids, kind="TPE", data_path=data_folder, iter_no='', atlas="cc200")
n_samples = X.shape[0]
accs = []
accs_weighted = []
for tgt in range(20):
    print('Target:', tgt)
    src_sites = [site for site in range(20) if site != tgt]
    tgt_idx = np.where(site_label == tgt)[0]
    src_idx = [site_label == i for i in range(20) if i != tgt]
    x_tgt = X[tgt_idx]
    n_tgt = x_tgt.shape[0]
    y_tgt = y[tgt_idx]
    # x_src = np.concatenate([X[idx_.reshape(-1)] for idx_ in src_idx], axis=0)
    # y_src = np.concatenate([y[idx_.reshape(-1)] for idx_ in src_idx], axis=0)
    best_dim = 100
    best_alpha = 1
    best_beta = 1
    best_score = 0
    estimator = None
    # for dim in dims:
    #     print('Dim:', dim)
    #     malrr = sio.loadmat(root_dir + 'MALRR_FEAT/target_%s_TPE_%s_1_1_malrr.mat' % (tgt, dim))
    #     w = malrr['W']
    #     Z = malrr['Z'][0]
    #     x_train = np.concatenate([np.linalg.multi_dot([Zi.T, x_tgt, w.T]) for Zi in Z], axis=0)
    #     grid_search = GridSearchCV(estimator=LinearSVC(), param_grid={'C': [1, 5, 10]}, cv=5)
    #     grid_search.fit(x_train, y_src)
    #     print('Best score:', grid_search.best_score_)
    #     if grid_search.best_score_ > best_score:
    #         best_score = grid_search.best_score_
    #         best_dim = dim
    #         estimator = grid_search.best_estimator_

    # for alpha in alphas:
    #     print('Alpha:', alpha)
    #     malrr = sio.loadmat(root_dir + 'MALRR_FEAT/target_%s_TPE_100_%s_1_malrr.mat' % (tgt, alpha))
    #     w = malrr['W']
    #     Z = malrr['Z'][0]
    #     x_train = np.concatenate([np.linalg.multi_dot([Zi.T, x_tgt, w.T]) for Zi in Z], axis=0)
    #     grid_search = GridSearchCV(estimator=LinearSVC(), param_grid={'C': [1, 5, 10]}, cv=5)
    #     grid_search.fit(x_train, y_src)
    #     print('Best score:', grid_search.best_score_)
    #     if grid_search.best_score_ > best_score:
    #         best_score = grid_search.best_score_
    #         best_alpha = alpha
    #         estimator = grid_search.best_estimator_

    for beta in betas:
        print('Beta:', beta)
        malrr = sio.loadmat(root_dir + 'MALRR_FEAT/target_%s_TPE_100_1_%s_malrr.mat' % (tgt, beta))
        w = malrr['W']
        Z = malrr['Z'][0]
        Wi = malrr['Wi'][0]
        Ez = malrr['Ez'][0]
        # x_train = np.concatenate([np.linalg.multi_dot([Zi.T, x_tgt, w.T]) for Zi in Z], axis=0)

        x_train = np.concatenate([np.dot(X[site_label == src_sites[j]], Wi[j].T) - Ez[j].T for j in range(19)], axis=0)
        y_train = np.concatenate([y[site_label == src_sites[j]] for j in range(19)], axis=0)
        grid_search = GridSearchCV(estimator=LinearSVC(), param_grid={'C': [1, 5, 10]}, cv=5)
        grid_search.fit(x_train, y_train)
        print('Best score:', grid_search.best_score_)
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_alpha = beta
            estimator = grid_search.best_estimator_

    malrr = sio.loadmat(root_dir + 'MALRR_FEAT/target_%s_TPE_%s_%s_%s_malrr.mat' % (tgt, best_dim, best_alpha,
                                                                                    best_beta))
    w = malrr['W']
    Z = malrr['Z'][0]
    Wi = malrr['Wi'][0]
    Ez = malrr['Ez'][0]
    x_train = np.concatenate([np.dot(X[site_label == src_sites[j]], Wi[j].T) - Ez[j].T for j in range(19)], axis=0)
    y_train = np.concatenate([y[site_label == src_sites[j]] for j in range(19)], axis=0)
    x_test = np.dot(x_tgt, w.T)
    y_pred = estimator.predict(x_test)
    acc = accuracy_score(y_tgt, y_pred)
    print('Accuracy:', acc)
    accs.append(acc)
    accs_weighted.append(acc * n_tgt / n_samples)

print('Mean accuracy:', np.mean(accs))
print('Mean weighted accuracy:', np.mean(accs_weighted))
