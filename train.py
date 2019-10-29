import cv2
import numpy as np
import pylab as plt
from glob import glob
import argparse
import os
#import progressbar
import pickle as pkl
from numpy.lib import stride_tricks
from skimage import feature
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
#import mahotas as mt
from scipy import stats
import imageio
from haralick_feat_gen import haralick_features
from grey_scale_bkgrnd_foregrnd_seg import img_grey_scale_preprocess
from imblearn.combine import SMOTETomek

def check_args(args):

    if not os.path.exists(args.image_dir):
        raise ValueError("Image directory does not exist")

    if not os.path.exists(args.label_dir):
        raise ValueError("Label directory does not exist")

    if args.classifier != "SVM" and args.classifier != "RF" and args.classifier != "GBC":
        raise ValueError("Classifier must be either SVM, RF or GBC")

    if args.output_model.split('.')[-1] != "p":
        raise ValueError("Model extension must be .p")

    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir" , help="Path to images", required=True)
    parser.add_argument("-l", "--label_dir", help="Path to labels", required=True)
    parser.add_argument("-c", "--classifier", help="Classification model to use", required = True)
    parser.add_argument("-o", "--output_model", help="Path to save model. Must end in .p", required = True)
    parser.add_argument('-h_lick_p',"--haralick_params",help="Path to json dictionary of haralick features",required=True)
    args = parser.parse_args()
    return check_args(args)

def read_data(image_dir, label_dir):
    """The purpose of this method is to read in the dataset for analysis"""
    print ('[INFO] Reading image data.')

    filelist = glob(os.path.join(image_dir, '*.jpg'))
    image_list = []
    label_list = []

    for file in filelist:
        
        img=imageio.imread(file)
        label=imageio.imread(os.path.join(label_dir, os.path.basename(file).split('.')[0]+'.png'))
        #Cropped image segmented using method. 
        img_crop,label_crop=img_grey_scale_preprocess(img,label,20,21)

        image_list.append(img_crop)
        label_list.append(label_crop)

    return image_list,label_list

def subsample(features, labels, low, high, sample_size):

    idx = np.random.randint(low, high, sample_size)

    return features[idx], labels[idx]

def subsample_idx(low, high, sample_size):

    return np.random.randint(low,high,sample_size)


def create_features(img, label, haralick_params,train=True):

    num_examples = 1000 # number of examples per image to use for training model
    #Determine the number of unique values present in the mask
    n_uniq_vals=np.unique(label)
    
    
    text_rast_img=None
    for param_sets in haralick_params:
        
        #Generating haralick features based on parameter setpoint 
        #Dynamic grey level setting placed into system based on unique grey levels present in the image. 
        texture_img = haralick_features(img,param_sets['window'],param_sets['offset'],
                                        param_sets['theta_angle'],256, #np.unique(img)[0].shape
                                        param_sets['props'])
        if text_rast_img is None:
            text_rast_img=texture_img
        else:
            #Stacking each raster via depth 
            text_rast_img=np.dstack((text_rast_img,texture_img))
        ipdb.set_trace()
        
    #Flattening image out into shappe method for analysis
    features = text_rast_img.reshape(text_rast_img.shape[0]*text_rast_img.shape[1],
                                     text_rast_img.shape[2])
    label_flat=label.reshape(label.shape[0]*label.shape[1],1)
    
    #Performing SMOTE TOMEK over under sampling to boost performance due to class imbalance. 
    smt = SMOTETomek(ratio='auto')
    features_smt, label_smt = smt.fit_sample(features, label_flat)
    #Class based subsampling required in order to get this system to operate effectively.
    if train == True:
        #Randomly sample from feature setpost class rebalancing usign SMOTE TOMEK process
        ss_idx = subsample_idx(0, features_smt.shape[0], num_examples)
        features = features_smt[ss_idx]
        labels = label_smt[ss_idx]
    else:
        ss_idx = []
        labels = None
        
    return features, labels

def create_training_dataset(image_list, label_list,haralick_param):

    print ('[INFO] Creating training dataset on %d image(s).' %len(image_list))

    X = []
    y = []

    for i, img in enumerate(image_list):

        features, labels = create_features(img,label_list[i],haralick_param)
        X.append(features)
        y.append(labels)

    X = np.array(X)
    #X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
    y = np.array(y)
    #y = y.reshape(y.shape[0]*y.shape[1], y.shape[2]).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    print ('[INFO] Feature vector size:', X_train.shape)

    return X_train, X_test, y_train, y_test

def train_model(X, y, classifier):

    if classifier == "SVM":
        from sklearn.svm import SVC
        print ('[INFO] Training Support Vector Machine model.')
        model = SVC(class_weights='balanced')
        model.fit(X, y)
    elif classifier == "RF":
        from sklearn.ensemble import RandomForestClassifier
        print ('[INFO] Training Random Forest model.')
        model = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42)
        model.fit(X, y)
    elif classifier == "GBC":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        model.fit(X, y)

    print ('[INFO] Model training complete.')
    print ('[INFO] Training Accuracy: %.2f' %model.score(X, y))
    return model

def test_model(X, y, model):

    pred = model.predict(X)
    precision = metrics.precision_score(y, pred, average='weighted', labels=np.unique(pred))
    recall = metrics.recall_score(y, pred, average='weighted', labels=np.unique(pred))
    f1 = metrics.f1_score(y, pred, average='weighted', labels=np.unique(pred))
    accuracy = metrics.accuracy_score(y, pred)

    print ('--------------------------------')
    print ('[RESULTS] Accuracy: %.2f' %accuracy)
    print ('[RESULTS] Precision: %.2f' %precision)
    print ('[RESULTS] Recall: %.2f' %recall)
    print ('[RESULTS] F1: %.2f' %f1)
    print ('--------------------------------')

def main(image_dir, label_dir, classifier, output_model,haralick_param):

    start = time.time()
    
    image_list, label_list = read_data(image_dir, label_dir)
    X_train, X_test, y_train, y_test = create_training_dataset(image_list, label_list,haralick_param)
    model = train_model(X_train, y_train, classifier)
    test_model(X_test, y_test, model)
    pkl.dump(model, open(output_model, "wb"))
    print ('Processing time:',time.time()-start)

if __name__ == "__main__":
    args = parse_args()
    image_dir = args.image_dir
    label_dir = args.label_dir
    classifier = args.classifier
    output_model = args.output_model
    
    with open(arg.h_lick_p,'r') as file_read:
        h_lick_param=json.load(file_read)    
    
    main(image_dir, label_dir, classifier, output_model,h_lick_param)
