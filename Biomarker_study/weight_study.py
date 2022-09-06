import os
import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from nilearn import connectome, plotting

import warnings
import time

sys.path.append('../')
# from imports.utils import str2bool
# from imports.preprocess_data import get_timeseries, get_ids
from imports import preprocess_data as reader
from imports.MIDA import MIDA
from imports.utils import arg_parse
from config import get_cfg_defaults

warnings.filterwarnings("ignore")


def main():
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

    root_dir = cfg.DATASET.ROOT
    data_folder = os.path.join(root_dir, cfg.DATASET.BASE_DIR)
    pheno_fpath = os.path.join(root_dir, cfg.DATASET.PHENO_FILE)
    out_path = cfg.OUTPUT.OUT_PATH

    pheno_info = pd.read_csv(pheno_fpath, index_col="SUB_ID", usecols=["SUB_ID", "SITE_ID", "DX_GROUP", "FILE_ID"])
    subject_ids = reader.get_ids(data_folder)
    pheno_ = pheno_info.loc[list(subject_ids.astype("int"))]
    labels = pheno_["DX_GROUP"].values
    data = reader.get_timeseries(subject_ids, atlas_name=atlas, data_path=data_folder)

    site_ids = pheno_['SITE_ID'].values
    sites = list(np.unique(site_ids))
    sites.append("all")
    coefs = []

    for site in sites:
        print("Site: {}".format(site))
        if site == "all":
            train_id = np.arange(len(data))
        else:
            train_id = np.where(pheno_['SITE_ID'] != site)[0]
            # test_id = np.where(pheno_['SITE_ID'] == site)[0]
        site_encode = OneHotEncoder(sparse=False).fit_transform(site_ids.reshape(-1, 1)[train_id])
        data_train = [data[i] for i in train_id]

        # TPE
        fname = os.path.join(out_path, "%s_tpe.npy" % site)
        if os.path.exists(fname):
            conn_vecs = np.load(fname)
        else:
            start = time.time()
            conn_measure = connectome.ConnectivityMeasure(kind='correlation')
            input_data = conn_measure.fit_transform(data_train)
            conn_measure = connectome.ConnectivityMeasure(kind='tangent', vectorize=True, discard_diagonal=False)
            conn_measure.fit(input_data)
            conn_vecs = conn_measure.transform(input_data)
            end = time.time()
            print("Time for TPE: {}".format(end - start))
            np.save(fname, conn_vecs)

        # MIDA
        fname = os.path.join(out_path, "%s_mida.npy" % site)
        if os.path.exists(fname):
            W = np.load(fname)
            Z = np.linalg.multi_dot([conn_vecs, conn_vecs.T, W])
        else:
            start = time.time()
            Z, W = MIDA(conn_vecs, D=site_encode, kernel="linear", augment=False, return_w=True, h=200)
            end = time.time()
            print("Time for MIDA: {}".format(end - start))
            np.save(fname, W)

        # SVM
        fname = os.path.join(out_path, "%s_coef.npy" % site)
        if os.path.exists(fname):
            coef_ = np.load(fname)
        else:
            start = time.time()
            clf = LogisticRegression(max_iter=10000)
            clf.fit(Z, labels[train_id])
            end = time.time()
            print("Time for SVM: {}".format(end - start))
            coef_ = clf.coef_.T
            np.save(fname, coef_)
        coef = np.linalg.multi_dot((conn_vecs.T, W, coef_))
        coefs.append(coef.reshape((1, -1)))

    coefs = np.concatenate(coefs, axis=0).T
    conn_measure = connectome.ConnectivityMeasure(kind='correlation')
    coef_corr = conn_measure.fit_transform(coefs.reshape((-1,) + coefs.shape))[0]
    fname = os.path.join(out_path, "weight_corr.npy")
    np.save(fname, coef_corr)
    corrs = coef_corr[np.triu_indices(21, 1)]
    print("Max correlation: {}".format(np.max(corrs)))
    print("Min correlation: {}".format(np.min(corrs)))
    print("Mean correlation: {}".format(np.mean(corrs)))
    print("Std correlation: {}".format(np.std(corrs)))
    plotting.plot_matrix(coef_corr, vmax=1, vmin=-1, colorbar=True)

    corr_freq = np.zeros(coefs.shape)
    for i in range(coefs.shape[1]):
        idx = np.abs(coefs[:, i]).argsort()[::-1]
        corr_freq[idx[:50], i] = 1

    corr_freq_ = np.sum(corr_freq, axis=1)
    print("Top coefficient shared in 21 modes: {}".format(np.where(corr_freq_ == 21)[0].shape[0]))
    print("Top coefficient shared in >=16 modes: {}".format(np.where(corr_freq_ >= 16)[0].shape[0]))
    print("Top coefficient shared in >=12 modes: {}".format(np.where(corr_freq_ >= 12)[0].shape[0]))
    print("Top coefficient shared in >=10 modes: {}".format(np.where(corr_freq_ >= 10)[0].shape[0]))

    overlaps = []
    for i in range(20):
        overlap_coef = np.multiply(corr_freq[:, i], corr_freq[:, 20])
        overlaps.append(np.where(overlap_coef != 0)[0].shape[0])
    print(np.mean(overlaps))
    print(np.std(overlaps))


if __name__ == '__main__':
    main()
