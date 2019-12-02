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
from sklearn.metrics import f1_score,precision_recall_fscore_support
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
    parser.add_argument("-txt", "--text_dir" , help="Path to images", required=False)
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
    per_img_f1_score=[]
    for idx,file in enumerate(filelist):
        f_b_name=os.path.basename(file)
        tmp_dict={}
        tmp_dict['file_name']=f_b_name
        print ('[INFO] Processing images:',f_b_name)
        #If file name already exists pass and process forward. 
        predictions,labls=gen_predictions(file,h_lick_p,args,model)

        #f1 score performance metrics 
        tmp_dict.update(calc_f1_scr(labls,predictions))
        per_img_f1_score.append(tmp_dict)
        #ipdb.set_trace()
        #record intermittent results for analysis to ensure per class information is not lost. 
        if idx%5==0:
            
            timestr = time.strftime("%Y%m%d-%H%M%S")
            with open(os.path.join(output_dir,'f1score_per_cls'+timestr+'.pickle'),'wb') as fb:
                pickle.dump(per_img_f1_score,fb)

def calc_f1_scr(labls,predictions,
                 label_dict={'background':0,'liver':63,'l_kidney':126,'r_kidney':189,'spleen':252}):
    
    #Get per image resultsPer class performance
    prec,recall,f1,sprt=precision_recall_fscore_support(labls,predictions,average=None,
                            labels=list(label_dict.values()))
    
    #Uniq labels across both predictions and labls
    labls_present=np.unique(np.concatenate((labls,predictions)))
    #Getting nan values if colour is not present i mask
    per_cls_f1_scr_dict={k:(x if x in labls_present else 'NaN') for k,x in label_dict.items()}
    #returning final values if x is present 
    per_cls_f1_scr_dict={k:(x if x=='NaN' else {'prec':prec[i],
                                                'recall':recall[i],
                                                'f1score':f1[i],
                                                'support ':sprt[i]}) for i,(k,x) in enumerate(per_cls_f1_scr_dict.items())}
    
    non_wght_dict_avrg={'non_weighted_average':f1_score(labls,predictions,average='macro')}
    
    per_cls_f1_scr_dict.update(non_wght_dict_avrg)
    
    return per_cls_f1_scr_dict

def gen_predictions(file,h_lick_p,args,model):
    
    f_b_name=os.path.basename(file)
    dst_f_path=os.path.join(output_dir,'mask_pred_'+f_b_name)
    
    #Get image and file
    tmp_img,tmp_labl=get_mask_n_img_f(file)
    #If file is present for prediction just read in and return else generate
    #texture features and image from scratch where required
    if os.path.isfile(dst_f_path):
        
        return imageio.imread(dst_f_path).flatten(),tmp_labl.flatten()
        
    else:
        #Generate features for model based on initial analysis. 
        features,labls = create_features(os.path.splitext(f_b_name)[0],
                                               tmp_img,
                                               tmp_labl,
                                               h_lick_p,
                                               args,
                                               train=False)
        #Minx max scaling same parameters taken from training script 
        features,_=min_max_scaling(features)
        
        predictions=compute_prediction(features, model)
        
        write_img_to_file(predictions,tmp_img,dst_f_path)
        
        return predictions.flatten(),labls

def write_img_to_file(predictions,tmp_img,dst_f_path):
    #Writing final image to file following prediction
    inference_img = predictions.reshape((tmp_img.shape[0], tmp_img.shape[1]))
    imageio.imwrite(dst_f_path,inference_img)

def get_mask_n_img_f(file):
    
   tmp_img=imageio.imread(file)
            #replace from left to right path to masks. for opening up labels image.  
   tmp_labl=imageio.imread('masks'.join(str(file).rsplit('images',1)))                
   
   return tmp_img,tmp_labl
   
def main(image_dir, model_path, output_dir,args):

    infer_images(image_dir, model_path, output_dir,args)

if __name__ == '__main__':
    args = parse_args()
    
    model_path = args.model_path
    output_dir = args.output_dir
    image_dir=args.image_dir
    main(image_dir,model_path, output_dir,args)
