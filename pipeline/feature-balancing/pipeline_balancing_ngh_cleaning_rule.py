"""
This pipeline aimed at creating balanced set using neighbourhood cleaning rule
approach.
"""

import os

import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize

from protoclass.data_management import DCEModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import StandardTimeNormalization

from protoclass.extraction import EnhancementSignalExtraction

from imblearn.under_sampling import NeighbourhoodCleaningRule

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_dce = 'DCE_reg_bspline'
# Define the path of the ground for the prostate
path_gt = ['GT_inv/prostate', 'GT_inv/pz', 'GT_inv/cg', 'GT_inv/cap']
# Define the label of the ground-truth which will be provided
label_gt = ['prostate', 'pz', 'cg', 'cap']
# Define the path to the normalization parameters
path_norm = '/data/prostate/pre-processing/lemaitre-2016-nov/norm-objects'
# Define the path where to store the data
path_store = '/data/prostate/balanced/lemaitre-2016-nov/ngh-clean-rule'
if not os.path.exists(path_store):
    os.makedirs(path_store)

# Generate the different path to be later treated
path_patients_list_dce = []
path_patients_list_gt = []
# Create the generator
id_patient_list = [name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name))]
for id_patient in id_patient_list:
    # Append for the DCE data
    path_patients_list_dce.append(os.path.join(path_patients, id_patient,
                                               path_dce))
    # Append for the GT data - Note that we need a list of gt path
    path_patients_list_gt.append([os.path.join(path_patients, id_patient, gt)
                                  for gt in path_gt])

# Load all the data once.
data = []
label = []
for idx_pat in range(len(id_patient_list)):
    print 'Read patient {}'.format(id_patient_list[idx_pat])

    # Load the testing data that correspond to the index of the LOPO
    # Create the object for the DCE
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_patients_list_dce[idx_pat])
    print 'Read the DCE data for the current patient ...'

    # Create the corresponding ground-truth
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt,
                               path_patients_list_gt[idx_pat])
    print 'Read the GT data for the current patient ...'

    # Load the approproate normalization object
    filename_norm = (id_patient_list[idx_pat].lower().replace(' ', '_') +
                     '_norm.p')
    dce_norm = StandardTimeNormalization.load_from_pickles(
        os.path.join(path_norm, filename_norm))

    dce_mod = dce_norm.normalize(dce_mod)

    # Create the object to extrac data
    dce_ese = EnhancementSignalExtraction(DCEModality())

    # Concatenate the training data
    data.append(dce_ese.transform(dce_mod, gt_mod, label_gt[0]))
    # Extract the corresponding ground-truth for the testing data
    # Get the index corresponding to the ground-truth
    roi_prostate = gt_mod.extract_gt_data('prostate', output_type='index')
    # Get the label of the gt only for the prostate ROI
    gt_cap = gt_mod.extract_gt_data('cap', output_type='data')
    label.append(gt_cap[roi_prostate])
    print 'Data and label extracted for the current patient ...'


# Create the balanced set for each patient already extracted
data_balanced = []
label_balanced = []
for idx_pat, (data_pat, label_pat) in enumerate(zip(data, label)):
    print 'Balanced the set #{}'.format(idx_pat+1)

    # Balanced the set
    ncr = NeighbourhoodCleaningRule()

    # Under-sample the data
    data_b, label_b = ncr.fit_transform(data[idx_pat], label[idx_pat])

    # Append the balanced data and label
    data_balanced.append(data_b)
    label_balanced.append(label_b)

# Store the unbalanced and balanced data and label
np.save(os.path.join(path_store, 'unb_data.npy'), data)
np.save(os.path.join(path_store, 'unb_label.npy'), label)
np.save(os.path.join(path_store, 'bal_data.npy'), data_balanced)
np.save(os.path.join(path_store, 'bal_label.npy'), label_balanced)
