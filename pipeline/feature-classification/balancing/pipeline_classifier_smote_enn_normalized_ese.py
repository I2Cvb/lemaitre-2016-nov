"""
This pipeline is used to classify enhacement signals have been normalized.
"""

import os

import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize

from protoclass.data_management import DCEModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import StandardTimeNormalization

from protoclass.extraction import EnhancementSignalExtraction

from protoclass.classification import Classify

# Define the path where all the patients are
path_data = '/data/prostate/balanced/lemaitre-2016-nov/smote-enn'
# Define the path where all the patients are
path_patients = '/data/prostate/experiments'

# Create the generator
id_patient_list = [name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name))]

# Load the balanced and unbalanced dataset
data_unbalanced = np.load(os.path.join(path_data, 'unb_data.npy'))
label_unbalanced = np.load(os.path.join(path_data, 'unb_label.npy'))
data_balanced = np.load(os.path.join(path_data, 'bal_data.npy'))
label_balanced = np.load(os.path.join(path_data, 'bal_label.npy'))

n_jobs = 48
config = [{'classifier_str': 'random-forest', 'n_estimators': 100,
           'gs_n_jobs': n_jobs},
          #{'classifier_str': 'knn', 'n_neighbors': 3, 'gs_n_jobs': n_jobs},
          #{'classifier_str': 'knn', 'n_neighbors': 5, 'gs_n_jobs': n_jobs},
          #{'classifier_str': 'knn', 'n_neighbors': 7, 'gs_n_jobs': n_jobs},
          {'classifier_str': 'naive-bayes', 'gs_n_jobs': n_jobs},
          {'classifier_str': 'logistic-regression', 'gs_n_jobs': n_jobs},
          {'classifier_str': 'linear-svm', 'gs_n_jobs' : n_jobs}]
          #{'classifier_str': 'kernel-svm', 'gs_n_jobs' : n_jobs}]

result_config = []
for c in config:
    print 'The following configuration will be used: {}'.format(c)

    result_cv = []
    # Go for LOPO cross-validation
    for idx_lopo_cv in range(len(id_patient_list)):

        # Display some information about the LOPO-CV
        print 'Round #{} of the LOPO-CV'.format(idx_lopo_cv + 1)

        # Get the testing data which come from the unbalanced set
        testing_data = data_unbalanced[idx_lopo_cv]
        testing_label = np.ravel(label_binarize(label_unbalanced[idx_lopo_cv],
                                                [0, 255]))
        print 'Create the testing set ...'

        # Create the training data and label which come from the balanced set
        training_data = [arr for idx_arr, arr in enumerate(data_balanced)
                         if idx_arr != idx_lopo_cv]
        training_label = [arr for idx_arr, arr in enumerate(label_balanced)
                         if idx_arr != idx_lopo_cv]
        # Concatenate the data
        training_data = np.vstack(training_data)
        training_label = np.ravel(label_binarize(
            np.hstack(training_label).astype(int),
            [0, 255]))
        print 'Create the training set ...'

        # Perform the classification for the current cv and the
        # given configuration
        result_cv.append(Classify(training_data, training_label,
                                  testing_data, testing_label,
                                  **c))

    # Concatenate the results per configuration
    result_config.append(result_cv)

# Save the information
path_store = '/data/prostate/results/lemaitre-2016-nov/smote_enn'
if not os.path.exists(path_store):
    os.makedirs(path_store)
joblib.dump(result_config, os.path.join(path_store,
                                        'results_normalized_ese.pkl'))
