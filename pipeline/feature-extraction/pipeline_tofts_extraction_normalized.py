"""
This pipeline is used to find the extract the Tofts parameters
and saved the data in a matrix.
"""

import os

import numpy as np

from joblib import Parallel, delayed

from protoclass.data_management import DCEModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import StandardTimeNormalization

from protoclass.extraction import ToftsQuantificationExtraction

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
# Define the path to the normalization parameters
path_norm = '/data/prostate/pre-processing/lemaitre-2016-nov/norm-objects'
# Define the path to store the Tofts data
path_store = '/data/prostate/pre-processing/lemaitre-2016-nov/tofts-features-normalized'

# Parameters for Tofts quantization
T10 = 1.6
CA = 3.5

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

# # TEMPORARY FOR RE-COMPUTING SOME PATIENTS
# path_patients_list_dce = path_patients_list_dce[10:]
# path_patients_list_gt = path_patients_list_gt[10:]
# id_patient_list = id_patient_list[10:]

for p_dce, p_gt, pat in zip(path_patients_list_dce, path_patients_list_gt,
                            id_patient_list):

    print 'Processing patient #{}'.format(pat)

    # Create the Tofts Extractor
    tofts_ext = ToftsQuantificationExtraction(DCEModality(), T10, CA)

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

    # Add a shift to all the data to not have any issue during fitting with
    # the exponential
    # Check if the minimum is negative
    if np.min(dce_mod.data_) < 0:
        dce_mod.data_ -= np.min(dce_mod.data_)
        dce_mod.update_histogram()

    # Fit the parameters for Tofts
    print 'Extract Tofts parameters'

    aif_params = [(2.07149091e+00, 2.17230025e+01),
                  (9.72270021e+00, 2.65991086e+00),
                  (1.20972869e+00, 5.55108421e+01),
                  9.48135519e-02,
                  8.92495798e-05,
                  3.63079733e-02,
                  6.12571259e+01,
                  3]
    tofts_ext.fit(dce_mod, ground_truth=gt_mod, cat=label_gt[0], fit_aif=False,
                  aif_params=aif_params)

    # Extract the matrix
    print 'Extract the feature matrix'
    data = tofts_ext.transform(dce_mod, ground_truth=gt_mod, cat=label_gt[0])

    pat_chg = pat.lower().replace(' ', '_') + '_tofts.npy'
    filename = os.path.join(path_store, pat_chg)
    np.save(filename, data)
