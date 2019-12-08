# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:28:29 2019

@author: niall
"""

import argparse 
import os 
import pandas as pd
from grey_scale_bkgrnd_foregrnd_seg import img_grey_scale_preprocess
from train import read_data
from statistics import mean,median
from scipy.stats import kurtosis,skew
from PIL import Image
import numpy as np
import imageio

def parse_args():
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-i','-input_dir',
                        help='Input directory for images that will be coarsely segmented',type=str,required=True)
    parser.add_argument('-o','-output_dir',
                        help='Output directory for images will be coursely segmented',type=str,required=True)
    
    return parser.parse_args()

def main(args):
    
    img_dict=read_data(args.input_dir)
    
    img_avrg_dim=det_avrg_img_size(img_dict)
    
    img_resize_dict=resize_imgs(img_dict,img_avrg_dim)
    
    write_output_img(img_resize_dict,args)
    
def write_output_img(img_dict,args):
    
    #Generating directory paths for analysis
    dir_pths_dict={x[:-1]:os.path.join(args.output_dir,x) for x in ['images','masks']}
    #Generating the directory for analysis
    {k:gen_dir(v) for k,v in dir_pths_dict.items}
    #Iterating through final list for analysis
    for f_nms,img_arrs in img_dict.items():
        #Iterating through per mask and image arrays for writing to output folders
        for img_type,img in img_arrs.items():
            img_path=os.path.join(dir_pths_dict[img_type],f_nms)
            imageio.imwrite(img_path,img)
    
def gen_dir(dir_pth:str):
    
    if not os.path.exists(dir_pth):
        os.mkdir(dir_pth)
    
def resize_img_dict(img_dict:dict,img_avrg_dim:tuple)->dict:
    """resizing each image based on pil resize function. """
    for k,v in img_dict.items():
        
        img_dict[k]=resize_img_pil(v,img_avrg_dim)
        
    return img_dict
        
def resize_img(imgs_dict:dict,img_avrg_dim)->np.ndarray:
    
    tmp_dict_pil={k:Image.fromarray(v) for k,v in imgs_dict.items()}
    
    tmp_dict_resz_pil={k:v.resize(img_avrg_dim) for k,v in tmp_dict_pil.items()}
    
    return {k:np.array(v) for k,v in tmp_dict_resz_pil.items()}
    
    
def det_avrg_img_size(img_dict:dict)->dict:
    
    
    img_dims=[k['image'].shape for k,v in img_dict.items]
  
    return eval_size_distribution(img_dims)
  
def eval_size_distribution(img_dim_lst,dim_vars=['x','y']):
    
    #Getting mean x and y value dimensions for analysis 
    dim_dict={k:[x[idx] for x in img_dim_lst] for idx,k in enumerate(dim_vars)}
    #basic normality check for using mean or median for each dimension
    norm_bool_dict={k:{'norm_dst':norm_dist_chk(v),
                      'val_lst':v} for k,v in dim_dict.items()}
    #Generate final dimension
    final_dim_dict={k:det_dim_avrg(**v) for k,v in norm_bool_dict}
    x_avrg=final_dim_dict['x']
    y_avrg=final_dim_dict['y']
    return x_avrg,y_avrg
    
def det_dim_avrg(norm_dst,val_dst):
    """The purpose of this method is to find either the mean or median of each attribute"""
    #If the distribution is determined to be normally distributed used the mean
    if norm_dst:
        return mean(val_dst)
    #If not with serious skew or kurtosis utilise the median value for analysis. 
    else:
        return median(val_dst)
    
    
def norm_dist_chk(lst_val:list)->bool:
    """Return a boolean for evaluation if file is not checking out"""
    tmp_arr=np.array(lst_val)
    tmp_kurt=kurtosis(tmp_arr)
    tmp_skew=skew(tmp_arr)
    #If Kurtoisis is greater than 3 
    if tmp_kurt>3 or tmp_skew>0.5:
        return False
    else:
        return True

if __name__=='__main__':
    
    arg_vals=parse_args()
    
    main(arg_vals)