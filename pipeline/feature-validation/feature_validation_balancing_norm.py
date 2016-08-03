"""
This pipeline is used to assess the performance of clustering centroids
sampling on normalized DCE data.
"""

import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy import interp

from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from protoclass.data_management import GTModality

from protoclass.validation import labels_to_sensitivity_specificity

# Define the filename where the results are stored
path_results = '/data/prostate/results/lemaitre-2016-nov'
path_balanced = ['cluster',
                 #'cnn',
                 'nearmiss-1',
                 'nearmiss-2',
                 'nearmiss-3',
                 'ngh-clean-rule',
                 'one-sided-sel',
                 'rnd-over-sampling',
                 'rnd-under-sampling',
                 'smote',
                 'smote-borderline-1',
                 'smote-borderline-2',
                 'smote_enn']
filename_results = 'results_normalized_ese.pkl'

# Define the name for display in the image
legend_balancing = ['Cluster Centroids',
                    #'Condensed Nearest Neighbours',
                    'NearMiss 1',
                    'NearMiss 2',
                    'NearMiss 3',
                    'Neighbourhood Cleaning Rule',
                    'One-Sided Selection',
                    'Random Over-Sampling',
                    'Random Under-Sampling',
                    'SMOTE',
                    'SMOTE Borderline 1',
                    'SMOTE Borderline 2',
                    'SMOTE + Edited Nearest Neighbours']

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
config = [{'classifier_str': 'random-forest', 'n_estimators': 100,
           'gs_n_jobs': n_jobs},
          #{'classifier_str': 'knn', 'n_neighbors': 3, 'gs_n_jobs': n_jobs},
          #{'classifier_str': 'knn', 'n_neighbors': 5, 'gs_n_jobs': n_jobs},
          #{'classifier_str': 'knn', 'n_neighbors': 7, 'gs_n_jobs': n_jobs},
          {'classifier_str': 'naive-bayes', 'gs_n_jobs': n_jobs},
          {'classifier_str': 'logistic-regression', 'gs_n_jobs': n_jobs}]
          #{'classifier_str': 'linear-svm', 'gs_n_jobs' : n_jobs}]
          #{'classifier_str': 'kernel-svm', 'gs_n_jobs' : n_jobs}]

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

# Load all the results
result_method_config = []
for bal_met in path_balanced:
    path_res = os.path.join(path_results, bal_met, filename_results)
    result_method_config.append(joblib.load(path_res))

# Iterate for a single classifier configuration
for idx_c, c in enumerate(config):

    # Create an handle for the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)


    # Iterate for all the different balancing method
    # for idx_met in [0]:
    for idx_met, method in enumerate(legend_balancing):

        print 'The method used is {}'.format(method)

        # Initialise a list for the sensitivity and specificity
        sens_list = []
        spec_list = []

        # Initilise the mean roc
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        # Create an handle for the figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for idx_cv in range(len(result_method_config[idx_met][idx_c])):

            # Print the information about the iteration in the cross-validation
            print 'Iteration #{} of the cross-validation'.format(idx_cv+1)

            # Compute some metrics
            sens, spec = labels_to_sensitivity_specificity(
                testing_label_config[idx_c][idx_cv],
                result_method_config[idx_met][idx_c][idx_cv][0])
            sens *= 100.
            spec *= 100.
            # Aggreagate the value to compute mean and std
            sens_list.append(sens)
            spec_list.append(spec)

            print 'Sensitivity = {} - Specificity = {}'.format(sens, spec)

            # Compute the mean ROC
            mean_tpr += interp(mean_fpr,
                               result_method_config[idx_met][idx_c][idx_cv][2].fpr,
                               result_method_config[idx_met][idx_c][idx_cv][2].tpr)
            mean_tpr[0] = 0.0

            # Plot the plot for this CV
            ax.plot(result_method_config[idx_met][idx_c][idx_cv][2].fpr,
                    result_method_config[idx_met][idx_c][idx_cv][2].tpr,
                    label='ROC fold #{} - AUC = {:1.4f}'.format(
                        idx_cv+1,
                        result_method_config[idx_met][idx_c][idx_cv][2].auc))


        print 'Mean and std sensitivity {} / {}'.format(np.mean(sens_list),
                                                        np.std(sens_list))
        print 'Mean and std specificity {} / {}'.format(np.mean(spec_list),
                                                        np.std(spec_list))

        # Plot the mean ROC
        mean_tpr /= len(result_method_config[idx_met][idx_c])
        mean_tpr[-1] = 1.0

        mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, 'k--',
                label='Mean ROC  - AUC = {:1.4f}'.format(mean_auc), lw=2)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')

        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='lower right',
                        bbox_to_anchor=(1.65, -0.1))
        # Save the plot
        plt.savefig('met_{}_{}.png'.format(idx_c, idx_met),
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight')




    #     gt_arr = np.array([])
    #     res_arr = np.array([])
    #     prob_arr = np.empty([0, 2])
    #     # Go for each cross-validation iteration
    #     for idx_cv in range(len(result_method_config[idx_met][idx_c])):

    #         # Print the information about the iteration in the cross-validation
    #         print 'Iteration #{} of the cross-validation'.format(idx_cv+1)

    #         # Concatenate the data
    #         gt_arr = np.append(gt_arr, testing_label_config[idx_c][idx_cv])
    #         res_arr = np.append(res_arr,
    #                             result_method_config[idx_met][idx_c][idx_cv][0])
    #         prob_arr = np.append(prob_arr,
    #                              result_method_config[idx_met][idx_c][idx_cv][1],
    #                              axis=0)

    #     # Compute some metrics
    #     sens, spec = labels_to_sensitivity_specificity(
    #         gt_arr,
    #         res_arr)
    #     sens *= 100.
    #     spec *= 100.

    #     print 'Sensitivity: {}, Specificity: {}'.format(sens, spec)

    #     fpr, tpr, thresholds = roc_curve(gt_arr, prob_arr[:, 1])
    #     roc_auc = roc_auc_score(gt_arr, prob_arr[:, 1])
    #     ax.plot(fpr, tpr,
    #             label='{} - AUC = {:1.4f}'.format(legend_balancing[idx_met],
    #                                               roc_auc), lw=2)

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve')

    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='lower right',
    #                     bbox_to_anchor=(1.65, -0.1))
    # # Save the plot
    # plt.savefig('results/roc/balacing_{}.png'.format(idx_c),
    #             bbox_extra_artists=(lgd,),
    #             bbox_inches='tight')
