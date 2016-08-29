"""
This pipeline is used to check the goodness of fit of the different models.
"""

import os

import numpy as np

path_root = '/data/prostate/pre-processing/lemaitre-2016-nov'
path_r2 = [
    os.path.join(path_root, 'brix-features-r2'),
    os.path.join(path_root, 'brix-features-normalized-r2'),
    os.path.join(path_root, 'hoffmann-features-r2'),
    os.path.join(path_root, 'hoffmann-features-normalized-r2'),
    os.path.join(path_root, 'pun-features-r2'),
    os.path.join(path_root, 'pun-features-normalized-r2'),
    os.path.join(path_root, 'semi-features-r2'),
    os.path.join(path_root, 'semi-features-normalized-r2'),
    os.path.join(path_root, 'tofts-features-r2'),
    os.path.join(path_root, 'tofts-features-normalized-r2'),
    os.path.join(path_root, 'tofts-features-patient-aif-r2'),
    os.path.join(path_root, 'tofts-features-patient-aif-normalized-r2')
       ]

# For each folder we can compute the mean r2
for p_r2 in path_r2:

    # Create where we will store all the value
    r2 = []
    # We have to open each npy file
    for root, dirs, files in os.walk(p_r2):

        # Create the string for the file to read
        for f in files:
            filename = os.path.join(root, f)

            # Open the file and get the last column
            data = np.load(filename)
            r2.append(data[:, -1])

    # Convert the list to an array
    r2 = np.concatenate(r2)
    # Find the index of the values to consider
    #idx = np.flatnonzero(np.bitwise_and(r2 < 1., r2 > 0.))
    idx = np.flatnonzero(np.bitwise_not(np.isnan(r2)))

    # Compute the mean and standard deviation of the coefficient
    # of determination
    mean_r2 = np.mean(r2[idx])
    std_r2 = np.std(r2[idx])

    print 'Coefficient of determination R2: mean of {} - std of {}'.format(
        mean_r2, std_r2)
