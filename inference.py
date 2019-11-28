import cv2
import numpy as np
import pylab as plt
from glob import glob
import argparse
import os
import pickle as pkl
from train import create_features,min_max_scaling
from joblib import load
import pathlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import imageio
import time
import pickle
import ipdb
import json

def check_args(args):
    
    if not os.path.exists(args.image_dir):
        raise ValueError("Image directory does not exist")

    if not os.path.exists(args.output_dir):
        raise ValueError("Output directory does not exist")

    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-txt", "--txt_image_dir" , help="Path to images", required=False)
    parser.add_argument("-h_lick_p","--haralick_params",help='path to json for haralick parameters',required=False)
    parser.add_argument("-i", "--image_dir" , help="Path to images", required=False)
    parser.add_argument("-m", "--model_path", help="Path to .p model", required=True)
    parser.add_argument("-o", "--output_dir", help="Path to output directory", required = True)
    args = parser.parse_args()
    return check_args(args)


def compute_prediction(img,model):
    
    
    predictions = model.predict(img)
    
    return predictions
def infer_images(image_dir, model_path, output_dir,args):

    filelist = list(pathlib.Path(os.path.join(image_dir,'images')).rglob('*.png'))

    print ('[INFO] Running inference on %s test images' %len(filelist))
    
    with open(args.haralick_params,'r') as fb:
        h_lick_p=json.load(fb)
    
    model = load(model_path)
    per_img_f1_score={}
    for idx,file in enumerate(filelist):
        f_b_name=os.path.basename(file)
        print ('[INFO] Processing images:',f_b_name)
        
        tmp_img=imageio.imread(file)
        #replace from left to right path to masks. for opening up labels image.  
        tmp_labl=imageio.imread('masks'.join(str(file).rsplit('images',1)))
        features,labls = create_features(os.path.splitext(f_b_name)[0],
                                               tmp_img,
                                               tmp_labl,
                                               h_lick_p,
                                               args.txt_image_dir,
                                               model_nm='N/A',
                                               train=False)
        #Minx max scaling same parameters taken from training script 
        features,_=min_max_scaling(features)
        
        predictions = compute_prediction(features, model)
        inference_img = predictions.reshape((tmp_img.shape[0], tmp_img.shape[1]))
        #f1 score performance metrics 
        #Per class performance
        
        
        #Get per image results
        per_img_f1_score[f_b_name]={'per_class':f1_score(labls,predictions,average=None),
                        'non_weighted_average':f1_score(labls,predictions,average='macro')}
        imageio.imwrite(os.path.join(output_dir,'mask_pred_'+f_b_name),inference_img)
        #record intermittent results for analysis to ensure per class information is not lost. 
        if idx%5==0:
            
            timestr = time.strftime("%Y%m%d-%H%M%S")
            with open(os.path.join(output_dir,'f1score_per_cls'+timestr+'.pickle'),'wb') as fb:
                pickle.dump(per_img_f1_score,fb)
                

def main(image_dir, model_path, output_dir,args):

    infer_images(image_dir, model_path, output_dir,args)

if __name__ == '__main__':
    args = parse_args()
    
    model_path = args.model_path
    output_dir = args.output_dir
    image_dir=args.image_dir
    main(image_dir,model_path, output_dir,args)
