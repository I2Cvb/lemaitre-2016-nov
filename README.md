A novel normalization approach for DCE-MRI applied to the prostate cancer detection
===================================================================================

How to use the pipeline?
------------------------

### Pre-processing pipeline

The following pre-processing routines were applied:

- Standard time normalization.

#### Data variables

In the file `pipeline/feature-preprocessing/pipeline_normalization_model.py`, you need to set the following variables:

- `path_patients`, `path_dce`, `path_gt`, `label_gt`: the appropriate data folder,
- `path_store_model`, `filename_model`: the path where to store the model computed.

In the file `pipeline/feature-preprocessing/pipeline_normalization_patient.py`, you need to set the following variables:

- `path_patients`, `path_dce`, `path_gt`, `label_gt`, `pt_mdl`: the appropriate data folder,
- `path_store`, `filename`: the path where to store the normalization object computed.

#### Algorithm variables

The variables that can be changed are:

- `params`: parameters of the standard time normalization. Refer to the help of `protoclass.StandardTimeNormalization` and the function `partial_fit_model` and `fit`.

#### Run the pipeline

From the root directory, launch an `ipython` or `python` prompt and run:

```
>> run pipeline/feature-preprocessing/pipeline_normalization_model.py
>> run pipeline/feature-preprocessing/pipeline_normalization_patient.py
```

### Classification pipeline

The following classification methods were applied:

- Random Forest,
- Naive-Bayes,
- Logistic Regression,
- k-NN,
- Linear-SVM,
- RBF-SVM.

#### Data variables

In the file `pipeline/feature-preprocessing/pipeline_classifier_unormalized_ese.py`, you need to set the following variables:

- `path_patients`, `path_dce`, `path_gt`, `label_gt`: the appropriate data folder,
- `path_store`: the path where to store the normalization object computed.

In the file `pipeline/feature-preprocessing/pipeline_classifier_normalized_ese.py`, you need to set the following variables:

- `path_patients`, `path_dce`, `path_gt`, `label_gt`, `path_norm`: the appropriate data folder,
- `path_store`: the path where to store the normalization object computed.

#### Algorithm variables

The variables that can be changed are:

- `config`: which corresponds to the different classification methods to test.

#### Run the pipeline

From the root directory, launch an `ipython` or `python` prompt and run:

```
>> run pipeline/feature-preprocessing/pipeline_classifier_unormalized_ese.py
>> run pipeline/feature-preprocessing/pipeline_classifier_normalized_ese.py
```
