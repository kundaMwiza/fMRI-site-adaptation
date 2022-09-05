import os
import scipy.io as sio
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure

import config
from imports import preprocess_data as reader
from imports.utils import arg_parse
from config import get_cfg_defaults


def main():
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    pipeline = cfg.DATASET.PIPELINE
    atlas = cfg.DATASET.ATLAS
    connectivity = cfg.METHOD.CONNECTIVITY
    download = cfg.DATASET.DOWNLOAD

    root_dir = cfg.DATASET.ROOT
    data_folder = os.path.join(root_dir, cfg.DATASET.BASE_DIR)
    out_path = cfg.OUTPUT.OUT_PATH
    files = ['rois_' + atlas]

    phenotype_file = os.path.join(root_dir, "ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    if download or not os.path.exists(phenotype_file):
        datasets.fetch_abide_pcp(data_dir=root_dir, pipeline=pipeline, band_pass_filtering=True,
                                 global_signal_regression=False, derivatives=files, quality_checked=cfg.DATASET.QC)

    phenotype_df = reader.get_phenotype(phenotype_file)

    file = open("cp_data.sh", "w")

    for sub_id in phenotype_df["SUB_ID"].values:
        subject_folder = os.path.join(data_folder, "%s" % sub_id)
        if os.path.exists(subject_folder):
            sub_file_path = os.path.join(subject_folder, "%s_%s_%s.mat" % (sub_id, atlas, connectivity))
            if os.path.exists(sub_file_path):
                file.write("cp %s %s\n" % (sub_file_path, out_path))

    file.close()


if __name__ == '__main__':
    main()
