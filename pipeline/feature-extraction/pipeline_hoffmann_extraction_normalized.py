"""
This pipeline is used to find the extract the Hoffmann parameters
and saved the data in a matrix.
"""

import os

import numpy as np

from joblib import Parallel, delayed

from protoclass.data_management import DCEModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import StandardTimeNormalization

from protoclass.extraction import BrixQuantificationExtraction

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_dce = 'DCE_reg_bspline'
# Define the path of the ground for the prostate
path_gt = 'GT_inv/prostate'
# Define the label of the ground-truth which will be provided
label_gt = ['prostate']
# Define the path to the normalization parameters
path_norm = '/data/prostate/pre-processing/lemaitre-2016-nov/norm-objects'
# Define the path to store the Tofts data
path_store = '/data/prostate/pre-processing/lemaitre-2016-nov/hoffmann-features-normalized'

shift = np.array([233.33757962, 239.33121019, 242.32802548, 243.32696391,
                  247.32271762, 296.5, 376.66851169, 443.50369004,
                  468.44218942, 476.42250923, 487.39544895, 501.36100861,
                  510.33886839, 517.32164822, 522.30934809, 533.28228782,
                  539.26752768, 539.26752768, 539.26752768, 540.26506765,
                  550.2404674, 557.22324723, 557.22324723, 556.22570726,
                  556.22570726, 555.22816728, 556.99363057, 556.99363057,
                  556.99363057, 556.99363057, 556.99363057, 556.99363057,
                  556.99363057, 556.99363057, 556.99363057, 557.992569,
                  558.99150743, 558.99150743, 557.992569, 556.99363057])

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
    path_patients_list_gt.append([os.path.join(path_patients, id_patient,
                                               path_gt)])

for p_dce, p_gt, pat in zip(path_patients_list_dce, path_patients_list_gt,
                            id_patient_list):

    print 'Processing #{}'.format(pat)

    # Create the Tofts Extractor
    brix_ext = BrixQuantificationExtraction(DCEModality())

    # Read the DCE
    print 'Read DCE images'
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(p_dce)

    # Read the GT
    print 'Read GT images'
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, p_gt)

    # Load the approproate normalization object
    filename_norm = (pat.lower().replace(' ', '_') +
                     '_norm.p')
    dce_norm = StandardTimeNormalization.load_from_pickles(
        os.path.join(path_norm, filename_norm))

    dce_mod = dce_norm.normalize(dce_mod)

    for idx in range(dce_mod.data_.shape[0]):
        dce_mod.data_[idx, :] += shift[idx]

    dce_mod.update_histogram()

    # Fit the parameters for Brix
    print 'Extract Hoffmann'
    brix_ext.fit(dce_mod, ground_truth=gt_mod, cat=label_gt[0])

    # Extract the matrix
    print 'Extract the feature matrix'
    data = brix_ext.transform(dce_mod, ground_truth=gt_mod, cat=label_gt[0],
                              kind='hoffmann')

    pat_chg = pat.lower().replace(' ', '_') + '_hoffmann.npy'
    filename = os.path.join(path_store, pat_chg)
    np.save(filename, data)
