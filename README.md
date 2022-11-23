# Machine Learning - Image Segmentation

Per pixel image segmentation using machine learning algorithms. Please refer to the requirements.txt file for environment requirements. Compatible with Python 3.6.

# Image pre processing
Note there is an image processing script 'course_seg_resize_imgs_only.py' which can be used with MRI and CT scan images that performs the following functions:

* Basic otsu thresholding for bounding box segmentation around the primary blob of interest in the foreground of the scan(e.g. the ROI in question). The utility of pre processing the images like this prior to training is that it reduces the class imbalance issue of 0 value background pixels when training the SVM and ensures the texture rasters are focused on the primary foreground image of interest. 
* Resizes all images to the most common image WxH dimensions within the patient population being evaluated. 

Note that the output of these pre processed images will need to be visually inspected by the user manually to confirm it is working correctly.   

# Training data generation

The final feature vector generated for training is 2-D matrix, RxC where R is the number of sample rows and C is the feature channels. 
* The maximum size of R is defined by WXHX[No of images used]x[SMOTE-TEK sample rebalancing if required]. 
* The number of channels is defined by the features vectors that are employed to train the SVM please review the section 'Feature channels' and 'Example Usage' below to see further details on how to define the channels that can be used. 

### Feature channels:
1. Spectral:

* * Grey level features. 

2. Texture Haralick features:

> > GLCM(Grey Level Co-occurence Matrices) are generated per the parameters defined in the json file provided by the user against the "h_lick_p" flag when training a new model with the 'train.py' script. Haralick parameters are then calculated from these GLCM's with the parameters for this again being defined in the same json file provided by the user. Refer to 'glcm_feature_params.json' for a template of how this can be generated. The specific haralick feature types that can be calculated from these GLCM's are:

* * Contrast
* * Dissimilarity
* * Homogeneity
* * Angular Second Moment
* * (ASM)
* * Energy
* * Correlation

> > Note that if haralick features have already been created for a given image set and are present in the "--text_dir" flag defined by the user these texture files will not be created again to ensure efficient texture file generation is performed. 


### Supported Learners

* Support Vector Machine using either a chi squared, nystrom non linear approximations or a linear kernel


### Example Usage
#### Training
Training each model in either a grid search pattern or with sepcified parameters given by the user can be performed with the following command:

```python 
train.py -i <path_to_image_folder> -l <path/to/label/folder> -c <SVM, RF, GBC> -o <path/to/model.p>
``` 

For further details on arguments please refer to the arg parse method "parse_args" present within the script. 

#### Inference
To leverage an existing model to generate prediction images for each image both in terms of binary prediction images and general analysis use the followig command:

python inference.py  -i <path_to_image_folder> -m <path/to/model.p> -o <path/to/output/folder> -txt <path/to/texture images (numpy format)> -h_lick_p <path/to/json file denoting haralick parameters> -p <path/to/patient substring file used to filter patient specific files for edge case analysis>

#### Visualisations: 
gen_analy_report_train_test_curves.py
Script used to generate training curves plots wrt to training set size and number of training iterations. 
  '-rw_r_dir':'source results directory where raw results are read in for analysis',
                        required=True)
  '-dst_dir':'destination directory to write output dataframes and plots from analysis',
  '-trn_data_dir':'source directory where original raw data is read in for additional training curve plot analysis',
  '-tst_data_dir':'source directory where raw test data is read in for additional curve plotting analysis',
  '-txt_params':'source directory to original json of texture features',
  '-t':'Destination path to texture feature files created during analysis'

### Requirements
train.py-imblearn API
inference.py:pandas_ml API  please note that if inference.py is being used imblearn needs to be hashed out of train.py due to the fact that there is requirements bug present between pandas-ml library and imblearn sklearn version requirements. Hence if either of these scripts are to be used an scikit-learn version <0.20. has to be used for pandas_ml in inference and >0.210 for imblearn for train.py prior to running either script.  
