"""
This pipeline is used to classify tofts parameters.
"""

import os

import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize

from protoclass.data_management import DCEModality
from protoclass.data_management import GTModality

from protoclass.extraction import ToftsQuantificationExtraction

from protoclass.classification import Classify

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the ground for the prostate
path_gt = ['GT_inv/prostate', 'GT_inv/pz', 'GT_inv/cg', 'GT_inv/cap']
# Define the label of the ground-truth which will be provided
label_gt = ['prostate', 'pz', 'cg', 'cap']
# Define the path to the normalization parameters
path_tofts = '/data/prostate/pre-processing/lemaitre-2016-nov/tofts-features-patient-aif'

# Generate the different path to be later treated
path_patients_list_gt = []
# Create the generator
id_patient_list = [name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name))]
for id_patient in id_patient_list:
    # Append for the GT data - Note that we need a list of gt path
    path_patients_list_gt.append([os.path.join(path_patients, id_patient, gt)
                                  for gt in path_gt])

# Load all the data once. Splitting into training and testing will be done at
# the cross-validation time
data = []
label = []
for idx_pat in range(len(id_patient_list)):
    print 'Read patient {}'.format(id_patient_list[idx_pat])

    # Create the corresponding ground-truth
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt,
                               path_patients_list_gt[idx_pat])
    print 'Read the GT data for the current patient ...'

    # Load the approproate normalization object
    filename_tofts = (id_patient_list[idx_pat].lower().replace(' ', '_') +
                      '_tofts.npy')

    # Concatenate the training data
    data.append(np.load(os.path.join(path_tofts, filename_tofts))[:, 1])
    # Extract the corresponding ground-truth for the testing data
    # Get the index corresponding to the ground-truth
    roi_prostate = gt_mod.extract_gt_data('prostate', output_type='index')
    # Get the label of the gt only for the prostate ROI
    gt_cap = gt_mod.extract_gt_data('cap', output_type='data')
    label.append(gt_cap[roi_prostate])
    print 'Data and label extracted for the current patient ...'

n_jobs = 48
config = [{'classifier_str': 'random-forest', 'n_estimators': 100,
           'gs_n_jobs': n_jobs},
          #{'classifier_str': 'knn', 'n_neighbors': 3, 'gs_n_jobs': n_jobs},
          #{'classifier_str': 'knn', 'n_neighbors': 5, 'gs_n_jobs': n_jobs},
          #{'classifier_str': 'knn', 'n_neighbors': 7, 'gs_n_jobs': n_jobs},
          {'classifier_str': 'naive-bayes', 'gs_n_jobs': n_jobs},
          {'classifier_str': 'logistic-regression', 'gs_n_jobs': n_jobs}]
          #{'classifier_str': 'linear-svm', 'gs_n_jobs' : n_jobs}]
          #{'classifier_str': 'kernel-svm', 'gs_n_jobs' : n_jobs}]

result_config = []
for c in config:
    print 'The following configuration will be used: {}'.format(c)

    result_cv = []
    # Go for LOPO cross-validation
    for idx_lopo_cv in range(len(id_patient_list)):

        # Display some information about the LOPO-CV
        print 'Round #{} of the LOPO-CV'.format(idx_lopo_cv + 1)

        # Get the testing data
        testing_data = np.atleast_2d(data[idx_lopo_cv]).T
        testing_label = label_binarize(label[idx_lopo_cv], [0, 255])
        print 'Create the testing set ...'

        # Create the training data and label
        training_data = [arr for idx_arr, arr in enumerate(data)
                         if idx_arr != idx_lopo_cv]
        training_label = [arr for idx_arr, arr in enumerate(label)
                         if idx_arr != idx_lopo_cv]
        # Concatenate the data
        training_data = np.atleast_2d(np.hstack(training_data)).T
        training_label = label_binarize(np.hstack(training_label).astype(int),
                                        [0, 255])
        print 'Create the training set ...'

        # Perform the classification for the current cv and the
        # given configuration
        result_cv.append(Classify(training_data, training_label,
                                  testing_data, testing_label,
                                  **c))

    # Concatenate the results per configuration
    result_config.append(result_cv)

# Save the information
path_store = '/data/prostate/results/lemaitre-2016-nov/tofts-ve-patient-aif'
if not os.path.exists(path_store):
    os.makedirs(path_store)
joblib.dump(result_config, os.path.join(path_store,
                                        'results_normalized_ese.pkl'))
