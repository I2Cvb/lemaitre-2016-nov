"""
This pipeline is used to find the extract the Weibull parameters
and saved the data in a matrix.
"""

import os

import numpy as np

from joblib import Parallel, delayed

from protoclass.data_management import DCEModality
from protoclass.data_management import GTModality

from protoclass.extraction import WeibullQuantificationExtraction

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_dce = 'DCE_reg_bspline'
# Define the path of the ground for the prostate
path_gt = 'GT_inv/prostate'
# Define the label of the ground-truth which will be provided
label_gt = ['prostate']
# Define the filename for the model
pt_mdl = '/data/prostate/pre-processing/lemaitre-2016-nov/model/model_stn.npy'
# Define the path to store the Tofts data
path_store = '/data/prostate/pre-processing/lemaitre-2016-nov/weibull-features'

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

    print 'Processing patient #{}'.format(pat)

    # Create the Tofts Extractor
    weibull_ext = WeibullQuantificationExtraction(DCEModality())

    # Read the DCE
    print 'Read DCE images'
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(p_dce)

    # Read the GT
    print 'Read GT images'
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, p_gt)

    # Fit the parameters for Brix
    print 'Extract Weibull'
    weibull_ext.fit(dce_mod, ground_truth=gt_mod, cat=label_gt[0])

    # Extract the matrix
    print 'Extract the feature matrix'
    data = weibull_ext.transform(dce_mod, ground_truth=gt_mod, cat=label_gt[0])

    pat_chg = pat.lower().replace(' ', '_') + '_weibull.npy'
    filename = os.path.join(path_store, pat_chg)
    np.save(filename, data)
