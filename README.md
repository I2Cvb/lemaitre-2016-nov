A novel normalization approach for DCE-MRI applied to the prostate cancer detection
===================================================================================

How to use the pipeline?
------------------------

### Data registration

#### Code compilation

Before to go to data mining and mahcine learning, there is a need to register the DCE-MRI data and the ground-truth.
The registration was programed in C++ with the ITK toolbox. You need to compile the code to be able to call the executable.

Therefore, you can compile the code from the root directory as:

```
$ mkdir build
$ cd build
$ cmake ../src
```

Two executables will be created in `bin/`:

- `reg_dce`: register DCE-MRI data to remove motion during the acquisition.
- `reg_gt`: register the T2W-MRI, DCE-MRI, and ground-truth data.

#### Run the executables

You can call the executable `./reg_dce` as:

```
./reg_dce arg1 arg2
```

- `arg1` is the folder with the DCE-MRI data,
- `arg2` is the storage folder with the registered DCE-MRI data.

You can call the executable `./reg_gt` as:

```
./reg_gt arg1 arg2 arg3 arg4
```

- `arg1` is the T2W-MRI ground-truth with the segmentation of the prostate.
- `arg2` is the DCE-MRI ground-truth with the segmentation of the prostate.
- `arg3` is the folder with the DCE-MRI with intra-modality motion correction (see `reg_dce`),
- `arg4` is the storage folder inter-modality motion correction.

### Pre-processing pipeline

The following pre-processing routines were applied:

- Standard time normalization.

Addionally, Tofts model requires an estimate of the AIF which can be derived from a population.

#### Data variables

In the file `pipeline/feature-preprocessing/pipeline_normalization_model.py`, you need to set the following variables:

- `path_patients`, `path_dce`, `path_gt`, `label_gt`: the appropriate data folder,
- `path_store_model`, `filename_model`: the path where to store the model computed.

In the file `pipeline/feature-preprocessing/pipeline_normalization_patient.py`, you need to set the following variables:

- `path_patients`, `path_dce`, `path_gt`, `label_gt`, `pt_mdl`: the appropriate data folder,
- `path_store`, `filename`: the path where to store the normalization object computed.

In the file `pipeline/feature-preprocessing/pipeline_average_aif.py` and `pipeline/feature-preprocessing/pipeline_average_aif_normalized.py`, you need to set the following variables:

- `path_patients`, `path_dce`, `path_gt`, `label_gt`: the appropriate data folder,

#### Algorithm variables

The variables that can be changed are:

- `params`: parameters of the standard time normalization. Refer to the help of `protoclass.StandardTimeNormalization` and the function `partial_fit_model` and `fit`.

#### Run the pipeline

To normalize the DCE MRI data, launch an `ipython` or `python` prompt and run from the root directory:

```python
>> run pipeline/feature-preprocessing/pipeline_normalization_model.py
>> run pipeline/feature-preprocessing/pipeline_normalization_patient.py
```

To compute the population-based AIF, launch an `ipython` or `python` prompt and run from the root directory:

```python
>> run pipeline/feature-preprocessing/pipeline_average_aif.py
>> run pipeline/feature-preprocessing/pipeline_average_normalized.py
```

These two scripts will provide an estimate of the AIF signal which will be used later in Tofts quantification.

### Extraction pipeline

The following quantification methods were applied:

- Brix model,
- Hoffmann model,
- PUN model,
- Tofts model,
- Semi-quantitative model.

#### Data variables

For each extraction method, the following variables need to be set:

- `path_patients`, `path_dce`, `path_gt`, `label_gt`: the appropriate data folder,
- `path_store`: the path where to store the normalization object computed.

Additionally, for the normalized data, the following variable need to be set:

- `path_norm`: the appropriate path to the normalization objects store in the previous section.

#### Algorithm variables

For the normalized data, you need to set:

- `shift`: the variable which is the maximum offtsets found during the preprocessing.

For the scripts `pipeline/feature-extraction/pipeline_tofts_extraction_population_aif***.py`, you need to set:

- `aif_params`: the parameters fo the population-based AIF found in pre-processing.

#### Run the pipeline

From the root directory, launch an `ipython` or `python` prompt and run:

```python
>> run pipeline/feature-extraction/pipeline_brix_extraction_normalized.py
>> run pipeline/feature-extraction/pipeline_brix_extraction.py
>> run pipeline/feature-extraction/pipeline_hoffmann_extraction_normalized.py
>> run pipeline/feature-extraction/pipeline_hoffmann_extraction.py
>> run pipeline/feature-extraction/pipeline_pun_extraction_normalized.py
>> run pipeline/feature-extraction/pipeline_pun_extraction.py
>> run pipeline/feature-extraction/pipeline_semi_quantitative_normalized.py
>> run pipeline/feature-extraction/pipeline_semi_quantitative.py
>> run pipeline/feature-extraction/pipeline_tofts_extraction_patient_aif_normalized.py
>> run pipeline/feature-extraction/pipeline_tofts_extraction_patient_aif.py
>> run pipeline/feature-extraction/pipeline_tofts_extraction_population_aif_normalized.py
>> run pipeline/feature-extraction/pipeline_tofts_extraction_population_aif.py
```

### Classification pipeline

The following classification methods were applied:

- Random Forest.

#### Data variables

For un-normalized data, you need to set the following variables:

- `path_patients`, `path_dce`, `path_gt`, `label_gt`: the appropriate data folder,
- `path_store`: the path where to store the normalization object computed.

For normalized data, you need to set the following variables:

- `path_patients`, `path_dce`, `path_gt`, `label_gt`, `path_norm`: the appropriate data folder,
- `path_store`: the path where to store the normalization object computed.

#### Algorithm variables

The variables that can be changed are:

- `config`: which corresponds to the different classification methods to test.

#### Run the pipeline

From the root directory, launch an `ipython` or `python` prompt and run:

```python
>> run pipeline/feature-classification/pipeline_classifier_unormalized_ese.py
>> run pipeline/feature-classification/pipeline_classifier_normalized_ese.py
>> # Brix model
>> run pipeline/feature-classification/brix/pipeline_classifier_brix_A_normalized.py
>> run pipeline/feature-classification/brix/pipeline_classifier_brix_A.py
>> run pipeline/feature-classification/brix/pipeline_classifier_brix_kel_normalized.py
>> run pipeline/feature-classification/brix/pipeline_classifier_brix_kel.py
>> run pipeline/feature-classification/brix/pipeline_classifier_brix_kep_normalized.py
>> run pipeline/feature-classification/brix/pipeline_classifier_brix_kep.py
>> run pipeline/feature-classification/brix/pipeline_classifier_brix_normalized.py
>> run pipeline/feature-classification/brix/pipeline_classifier_brix.py
>> # Hoffmann model
>> run pipeline/feature-classification/hoffmann/pipeline_classifier_hoffmann_A_normalized.py
>> run pipeline/feature-classification/hoffmann/pipeline_classifier_hoffmann_A.py
>> run pipeline/feature-classification/hoffmann/pipeline_classifier_hoffmann_kel_normalized.py
>> run pipeline/feature-classification/hoffmann/pipeline_classifier_hoffmann_kel.py
>> run pipeline/feature-classification/hoffmann/pipeline_classifier_hoffmann_kep_normalized.py
>> run pipeline/feature-classification/hoffmann/pipeline_classifier_hoffmann_kep.py
>> run pipeline/feature-classification/hoffmann/pipeline_classifier_hoffmann_normalized.py
>> run pipeline/feature-classification/hoffmann/pipeline_classifier_hoffmann.py
>> # PUN model
>> run pipeline/feature-classification/pun/pipeline_classifier_pun_a0_normalized.py
>> run pipeline/feature-classification/pun/pipeline_classifier_pun_a0.py
>> run pipeline/feature-classification/pun/pipeline_classifier_pun_beta_normalized.py
>> run pipeline/feature-classification/pun/pipeline_classifier_pun_beta.py
>> run pipeline/feature-classification/pun/pipeline_classifier_pun_normalized.py
>> run pipeline/feature-classification/pun/pipeline_classifier_pun.py
>> run pipeline/feature-classification/pun/pipeline_classifier_pun_r_normalized.py
>> run pipeline/feature-classification/pun/pipeline_classifier_pun_r.py
>> # Semi-quantitative model
>> run pipeline/feature-classification/semi/pipeline_classifier_semi_quantitative_auc_normalized.py
>> run pipeline/feature-classification/semi/pipeline_classifier_semi_quantitative_auc.py
>> run pipeline/feature-classification/semi/pipeline_classifier_semi_quantitative_normalized.py
>> run pipeline/feature-classification/semi/pipeline_classifier_semi_quantitative.py
>> run pipeline/feature-classification/semi/pipeline_classifier_semi_quantitative_rel_enh_normalized.py
>> run pipeline/feature-classification/semi/pipeline_classifier_semi_quantitative_rel_enh.py
>> run pipeline/feature-classification/semi/pipeline_classifier_semi_quantitative_tau_normalized.py
>> run pipeline/feature-classification/semi/pipeline_classifier_semi_quantitative_tau.py
>> run pipeline/feature-classification/semi/pipeline_classifier_semi_quantitative_washin_normalized.py
>> run pipeline/feature-classification/semi/pipeline_classifier_semi_quantitative_washin.py
>> run pipeline/feature-classification/semi/pipeline_classifier_semi_quantitative_washout_normalized.py
>> run pipeline/feature-classification/semi/pipeline_classifier_semi_quantitative_washout.py
>> # Tofts model
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_ktrans_patient_aif_normalized.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_ktrans_patient_aif.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_ktrans_population_aif_normalized.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_ktrans_population_aif.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_patient_aif_normalized.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_patient_aif.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_population_aif_normalized.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_population_aif.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_ve_patient_aif_normalized.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_ve_patient_aif.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_ve_population_aif_normalized.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_ve_population_aif.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_vp_patient_aif_normalized.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_vp_patient_aif.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_vp_population_aif_normalized.py
>> run pipeline/feature-classification/tofts/pipeline_classifier_tofts_vp_population_aif.py
```

### Validation pipeline

#### Run the pipeline

To generate the figues in the `results` directory, launch an `ipython` or `python` prompt and run from the root directory:

```python
>> run pipeline/feature-validation/feature_validation_normalisation_classifier.py
>> run pipeline/feature-validation/feature_validation_normalized_pharmaco_params.py
>> run pipeline/feature-validation/feature_validation_unormalized_pharmaco_params.py
>> run pipeline/feature-validation/feature_validation_all_methods_params.py
```
