"""
This pipeline is used to report the results for the different quantification
methods wihtout normalization.
"""

import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set(style='ticks', palette='Set2')
current_palette = sns.color_palette()

from scipy import interp

from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

from protoclass.data_management import GTModality

from protoclass.validation import labels_to_sensitivity_specificity

# Define the filename where the results are stored
path_results = '/data/prostate/results/lemaitre-2016-nov'
# Create a list with the different folder to check
# Brix
# Hoffmann
# Semi-quatitative
# Tofts
# Tofts patient AIF
filename_results = [
    os.path.join(path_results, 'brix-A/results_normalized_ese.pkl'),
    os.path.join(path_results, 'brix-kel/results_normalized_ese.pkl'),
    os.path.join(path_results, 'brix-kep/results_normalized_ese.pkl'),
    os.path.join(path_results, 'brix-A-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'brix-kel-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'brix-kep-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'hoffmann-A/results_normalized_ese.pkl'),
    os.path.join(path_results, 'hoffmann-kel/results_normalized_ese.pkl'),
    os.path.join(path_results, 'hoffmann-kep/results_normalized_ese.pkl'),
    os.path.join(path_results, 'hoffmann-A-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'hoffmann-kel-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'hoffmann-kep-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'pun-a0/results_normalized_ese.pkl'),
    os.path.join(path_results, 'pun-r/results_normalized_ese.pkl'),
    os.path.join(path_results, 'pun-beta/results_normalized_ese.pkl'),
    os.path.join(path_results, 'pun-a0-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'pun-r-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'pun-beta-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'semi-washin/results_normalized_ese.pkl'),
    os.path.join(path_results, 'semi-washout/results_normalized_ese.pkl'),
    os.path.join(path_results, 'semi-auc/results_normalized_ese.pkl'),
    os.path.join(path_results, 'semi-tau/results_normalized_ese.pkl'),
    os.path.join(path_results, 'semi-relenh/results_normalized_ese.pkl'),
    os.path.join(path_results, 'semi-washin-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'semi-washout-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'semi-auc-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'semi-tau-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'semi-relenh-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'tofts-ktrans/results_normalized_ese.pkl'),
    os.path.join(path_results, 'tofts-ve/results_normalized_ese.pkl'),
    os.path.join(path_results, 'tofts-vp/results_normalized_ese.pkl'),
    os.path.join(path_results, 'tofts-ktrans-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'tofts-ve-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'tofts-vp-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'tofts-ktrans-patient-aif/results_normalized_ese.pkl'),
    os.path.join(path_results, 'tofts-ve-patient-aif/results_normalized_ese.pkl'),
    os.path.join(path_results, 'tofts-vp-patient-aif/results_normalized_ese.pkl'),
    os.path.join(path_results, 'tofts-ktrans-patient-aif-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'tofts-ve-patient-aif-normalized/results_normalized_ese.pkl'),
    os.path.join(path_results, 'tofts-vp-patient-aif-normalized/results_normalized_ese.pkl')

    ]

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the ground for the prostate
path_gt = ['GT_inv/prostate', 'GT_inv/pz', 'GT_inv/cg', 'GT_inv/cap']
# Define the label of the ground-truth which will be provided
label_gt = ['prostate', 'pz', 'cg', 'cap']

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
label = []
for idx_pat in range(len(id_patient_list)):
    print 'Read patient {}'.format(id_patient_list[idx_pat])

    # Create the corresponding ground-truth
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt,
                               path_patients_list_gt[idx_pat])
    print 'Read the GT data for the current patient ...'

    # Extract the corresponding ground-truth for the testing data
    # Get the index corresponding to the ground-truth
    roi_prostate = gt_mod.extract_gt_data('prostate', output_type='index')
    # Get the label of the gt only for the prostate ROI
    gt_cap = gt_mod.extract_gt_data('cap', output_type='data')
    label.append(gt_cap[roi_prostate])
    print 'Data and label extracted for the current patient ...'

n_jobs = 48
config = [{'classifier_str': 'random-forest'}]

testing_label_config = []
for c in config:
    print 'The following configuration will be used: {}'.format(c)

    testing_label_cv = []
    # Go for LOPO cross-validation
    for idx_lopo_cv in range(len(id_patient_list)):

        # Display some information about the LOPO-CV
        print 'Round #{} of the LOPO-CV'.format(idx_lopo_cv + 1)

        testing_label = np.ravel(label_binarize(label[idx_lopo_cv], [0, 255]))
        testing_label_cv.append(testing_label)

    testing_label_config.append(testing_label_cv)

# Go trought the differen methods
# Create an handle for the figure
fig = plt.figure()
ax = fig.add_subplot(111)

method_tpr_mean = []
method_tpr_std = []
method_auc = []
method_auc_std = []

for fresults in filename_results:
    # Load the results
    result_config = joblib.load(fresults)

    config_tpr_mean = []
    config_tpr_std = []
    config_auc = []
    config_auc_std = []

    print fresults

    # Go for each configuration
    for idx_c, (result_cv, gt_label_cv) in enumerate(zip(result_config,
                                                         testing_label_config)):

        # Print the information about the predictor
        print 'The current configuration is: {}'.format(config[idx_c])

        # Initilise the mean roc
        mean_tpr = []
        mean_fpr = np.linspace(0, 1, 30)
        auc_pat = []

        # Go for each cross-validation iteration
        for idx_cv, (res_itr, gt_itr) in enumerate(zip(result_cv, gt_label_cv)):

            # Compute the mean ROC
            mean_tpr.append(interp(mean_fpr, res_itr[2].fpr, res_itr[2].tpr))
            mean_tpr[idx_cv][0] = 0.0
            auc_pat.append(auc(mean_fpr, mean_tpr[-1]))

        avg_tpr = np.mean(mean_tpr, axis=0)
        std_tpr = np.std(mean_tpr, axis=0)
        avg_tpr[-1] = 1.0
        config_tpr_mean.append(avg_tpr)
        config_tpr_std.append(std_tpr)
        config_auc.append(auc(mean_fpr, avg_tpr))
        config_auc_std.append(np.std(auc_pat))
        print 'AUC = {:1.3f} +- {:1.3f}'.format(auc(mean_fpr, avg_tpr),
                                                config_auc_std[-1])

    method_tpr_mean.append(config_tpr_mean)
    method_tpr_std.append(config_tpr_std)
    method_auc.append(config_auc)
    method_auc_std.append(config_auc_std)
