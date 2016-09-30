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

testing_label_cv = []
# Go for LOPO cross-validation
for idx_lopo_cv in range(len(id_patient_list)):

    # Display some information about the LOPO-CV
    print 'Round #{} of the LOPO-CV'.format(idx_lopo_cv + 1)

    testing_label = np.ravel(label_binarize(label[idx_lopo_cv], [0, 255]))
    testing_label_cv.append(testing_label)

fresults = '/data/prostate/results/lemaitre-2016-nov/sparse-ksvd/results_normalized_ese.pkl'
results = joblib.load(fresults)

# Define the different parameters for the K-SVD projection
dict_size = [100, 200, 300]
sp_level = [2, 4, 8, 16]

# Create the dictionary lists
dict_tpr_mean = []
dict_tpr_std = []
dict_auc = []
dict_auc_std = []
for d_sz in range(len(dict_size)):

    # Create the sparsity lists
    sp_tpr_mean = []
    sp_tpr_std = []
    sp_auc = []
    sp_auc_std = []
    for sp in range(len(sp_level)):

        # # Initialise a list for the sensitivity and specificity
        # Initilise the mean roc
        mean_tpr = []
        mean_fpr = np.linspace(0, 1, 30)
        auc_pat = []

        # Go for each cross-validation iteration
        for idx_cv in range(len(testing_label_cv)):

            # Print the information about the iteration in the cross-validation
            print 'Iteration #{} of the cross-validation'.format(idx_cv+1)

            # Compute the mean ROC
            mean_tpr.append(interp(mean_fpr,
                                   results[d_sz][sp][idx_cv][2].fpr,
                                   results[d_sz][sp][idx_cv][2].tpr))
            mean_tpr[idx_cv][0] = 0.0
            auc_pat.append(auc(mean_fpr, mean_tpr[-1]))

        avg_tpr = np.mean(mean_tpr, axis=0)
        std_tpr = np.std(mean_tpr, axis=0)
        avg_tpr[-1] = 1.0
        sp_tpr_mean.append(avg_tpr)
        sp_tpr_std.append(std_tpr)
        sp_auc.append(auc(mean_fpr, avg_tpr))
        sp_auc_std.append(np.std(auc_pat))

    dict_tpr_mean.append(sp_tpr_mean)
    dict_tpr_std.append(sp_tpr_std)
    dict_auc.append(sp_auc)
    dict_auc_std.append(sp_auc_std)

# Make a plot for each level of sparsity
for sp in range(len(sp_level)):
    # Create an handle for the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for d_sz in range(len(dict_size)):
        ax.plot(mean_fpr, dict_tpr_mean[d_sz][sp],
                label=str(dict_size[d_sz]) + r' atoms - AUC $= {:1.3f} \pm {:1.3f}$'.format(
                    dict_auc[d_sz][sp], dict_auc_std[d_sz][sp]),
                lw=2)
        ax.fill_between(mean_fpr,
                        dict_tpr_mean[d_sz][sp]+dict_tpr_std[d_sz][sp],
                        dict_tpr_mean[d_sz][sp]-dict_tpr_std[d_sz][sp],
                        facecolor=current_palette[d_sz], alpha=0.2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(r'ROC curve for sparsity level $\lambda={}$'.format(sp_level[sp]))

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='lower right')#,
                    #bbox_to_anchor=(1.4, 0.1))
    # Save the plot
    plt.savefig('results/sparse_level_{}.pdf'.format(sp_level[sp]),
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')
