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
import argparse
from imports import preprocess_data as reader
from imports.utils import str2bool
import os
import shutil
import sys

# Input data variables
# root_folder = '/path/to/data/'
# root_folder = "/media/shuo/MyDrive/data/brain"
root_folder = "D:/ML_data/brain/qc"
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal/')


def main():
    parser = argparse.ArgumentParser(description='Download ABIDE data and compute functional connectivity matrices')
    parser.add_argument('--pipeline', default='cpac', type=str, help='Pipeline to preprocess ABIDE data. Available '
                                                                     'options are ccs, cpac, dparsf and niak. default: '
                                                                     'cpac.')
    parser.add_argument('--atlas', default='cc200', help='Brain parcellation atlas. Options: ho, cc200 and cc400, '
                                                         'default: cc200.')
    parser.add_argument('--connectivity', default='correlation', type=str, help='Type of connectivity used for network '
                                                                                'construction options: correlation, '
                                                                                'partial correlation, covariance, '
                                                                                'tangent, TPE. Default: correlation.')
    parser.add_argument('--download', default=True, type=str2bool, help='Dowload data or just compute functional '
                                                                        'connectivity. default: True')
    args = parser.parse_args()
    print(args)

    params = dict()

    pipeline = args.pipeline
    atlas = args.atlas
    connectivity = args.connectivity
    download = args.download

    # Files to fetch

    files = ['rois_' + atlas]

    filemapping = {'func_preproc': 'func_preproc.nii.gz',
                   files[0]: files[0] + '.1D'}

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    shutil.copyfile('./subject_ids.txt', os.path.join(data_folder, 'subject_ids.txt'))

    # Download database files
    if download:
        datasets.fetch_abide_pcp(data_dir=root_folder, pipeline=pipeline, band_pass_filtering=True,
                                 global_signal_regression=False, derivatives=files, quality_checked=True)

    subject_ids = reader.get_ids()
    subject_ids = subject_ids.tolist()

    # Create a folder for each subject
    for s, fname in zip(subject_ids, reader.fetch_filenames(subject_ids, files[0], atlas)):
        subject_folder = os.path.join(data_folder, s)
        if not os.path.exists(subject_folder):
            os.mkdir(subject_folder)

        # Get the base filename for each subject
        base = fname.split(files[0])[0]

        # Move each subject file to the subject folder
        for fl in files:
            if not os.path.exists(os.path.join(subject_folder, base + filemapping[fl])):
                shutil.move(base + filemapping[fl], subject_folder)

    time_series = reader.get_timeseries(subject_ids, atlas)

    # Compute and save connectivity matrices
    # if connectivity in ['correlation', 'partial correlation', 'covariance']:
    if connectivity in ["correlation", 'partial correlation', 'covariance', 'tangent', "TPE"]:
        reader.subject_connectivity(time_series, subject_ids, atlas, connectivity)


if __name__ == '__main__':
    main()
