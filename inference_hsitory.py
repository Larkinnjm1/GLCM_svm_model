# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:59:08 2019

@author: aczd087
"""

    blnk_mask=np.where(trl_dict['generators']['mask'][i,:,:]>0,i,blnk_mask)
    
plt.imshow(blnk_mask)
trl_dict['id']
pwd 
cd Dataset/
cd masks/
trl_img=imageio.imread('JPCLN059.png')
import imageio
trl_img=imageio.imread('JPCLN059.png')
plt.imshow(trl_img)
from PIL import Image
tlr_meth=Image.fromarray(trl_img)
trl_img_resize=tlr_meth.resize((128,128))
trl_img_resize
trl_img_resize_arr=np.array(trl_img_resize)
plt.imshow(trl_img_resize_arr)
plt.imshow(blnk_mask)
trl_dict_2=data_tmp.get_next()
ls
cd ..
cd .
cd ..
trl_dict_2=data_tmp.get_next()
trl_dict_2.keys()
trl_dict_2['id']
trl_mask_2=trl_dict['generators']['mask']
plt.imshow(trl_mask_2)
plt.imshow(np.squeeze(trl_mask_2,axis=0))
for i in range(1,trl_dict['generators']['mask'].shape[0]-1):
    blnk_mask_2=np.where(trl_dict['generators']['mask'][i,:,:]>0,i,blnk_mask)
    
for i in range(1,trl_dict['generators']['mask'].shape[0]-1):
    blnk_mask_2=np.where(trl_mask_2[i,:,:]>0,i,blnk_mask)
    
plt.imshow(blnk_mask_2)
blnk_mask_2=np.zeros((trl_dict['generators']['mask'].shape[1:]))
for i in range(1,trl_dict['generators']['mask'].shape[0]-1):
    blnk_mask_2=np.where(trl_mask_2[i,:,:]>0,i,blnk_mask_2)
    
plt.imshow(blnk_mask_2)
trl_mask_2=trl_dict_2['generators']['mask']
blnk_mask_2=np.zeros((trl_dict['generators']['mask'].shape[1:]))
for i in range(1,trl_dict['generators']['mask'].shape[0]-1):
    blnk_mask_2=np.where(trl_mask_2[i,:,:]>0,i,blnk_mask_2)
    
plt.imshow(blnk_mask_2)
cd Dataset/masks/
trl_img=imageio.imread('JPCLN015')
trl_img=imageio.imread('JPCLN015.png')
tlr_meth_2=Image.fromarray(trl_img)
trl_img_resize_2=tlr_meth_2.resize((128,128))
trl_img_resize_arr_2=np.array(trl_img_resize_2)
plt.imshow(trl_img_resize_arr_2)
trl_img_2=trl_dict_2['generators']['image']
trl_img_2=trl_dict_2['generators'].keys()
trl_img_2
trl_img_2=trl_dict_2['generators']['data']
trl_img_2.shape
plt.imshow(np.squeeze(trl_img_2,axis=0))
ls
cd ../..
lls
ls
runfile('U:/Matwo-CapsNet/dataset_shape_analysis.py', wdir='U:/Matwo-CapsNet')
trl_ds=dataset.dataset_train()
trl_ds.keys()
trl_unaug=trl_ds.get_next()
tlr_unaug.keys()
trl_unaug.keys()
plt.imshow(np.squeeze(trl_unaug['generators']['data'],axis=0))
plt.imshow(np.squeeze(trl_unaug['generators']['mask'][0,:,:],axis=0))
plt.imshow(np.squeeze(trl_unaug['generators']['mask'][1,:,:],axis=0))
tmp_mask_sec=trl_unaug['generators']['mask'][1,:,:]
tmp_mask_sec.shape
plt.imshow(trl_unaug['generators']['mask'][1,:,:])
runfile('U:/Matwo-CapsNet/dataset_shape_analysis.py', wdir='U:/Matwo-CapsNet')
trl_ds=dataset.dataset_train()
trl_unaug=trl_ds.get_next()
runfile('U:/Matwo-CapsNet/dataset_shape_analysis.py', wdir='U:/Matwo-CapsNet')
trl_ds=dataset.dataset_train()
trl_unaug=trl_ds.get_next()
runfile('U:/Matwo-CapsNet/dataset_shape_analysis.py', wdir='U:/Matwo-CapsNet')
trl_ds=dataset.dataset_train()
trl_unaug=trl_ds.get_next()
runfile('U:/Matwo-CapsNet/dataset_shape_analysis.py', wdir='U:/Matwo-CapsNet')
trl_ds=dataset.dataset_train()
trl_unaug=trl_ds.get_next()
runfile('U:/Matwo-CapsNet/dataset_shape_analysis.py', wdir='U:/Matwo-CapsNet')
trl_ds=dataset.dataset_train()
trl_unaug=trl_ds.get_next()
plt.imshow(trl_unaug['generator']['data'])
trl_unaug.keys()
trl_unaug['generators']
plt.imshow(trl_unaug['generators']['data'])
plt.imshow(np.squeeze(trl_unaug['generators']['data'],axis=0))
runfile('U:/Matwo-CapsNet/dataset_shape_analysis.py', wdir='U:/Matwo-CapsNet')
trl_ds=dataset.dataset_train()
trl_unaug=trl_ds.get_next()
plt.imshow(np.squeeze(trl_unaug['generators']['data'],axis=0))
plt.imshow(trl_unaug['generators']['mask'][0,:,:])

## ---(Mon Nov 25 12:04:40 2019)---
cd C:\Users\aczd087\Downloads
from joblib import dump, load
tmp_model=load('SVM_f1_weighted_20191117-010510_best_model')
import glob
import Path
import pathlib
trl_text_imgs=pathlib.Path.rglob(r'F:\Biomedical images\Texture_images\texture_imgs_svm_training\texture_imgs_raw\*.npy')
trl_text_imgs=pathlib.Path(r'F:\Biomedical images\Texture_images\texture_imgs_svm_training\texture_imgs_raw').rglob('*.npy')
trl_text_imgs
list(trl_text_imgs)
text_imgs=list(trl_text_imgs)
import imageio
import numpy as np
trl_img=np.load(text_imgs[0])
text_imgs[0]
len(text_imgs)
text_imgs
trl_text_imgs=list(pathlib.Path(r'F:\Biomedical images\Texture_images\texture_imgs_svm_training\texture_imgs_raw').rglob('*.npy'))
trl_text_imgs[0]
trl_img=np.load(text_imgs[0])
trl_img=np.load(trl_text_imgs[0])
trl_img.shape
trl_img_reshape=trl_img.reshape(trl_img.shape[0]*trl_img.shape[1],trl_img.shape[2])
trl_img_reshape[0,:]
trl_img[0,0,:]==trl_img_reshape[0,:]
trl_img[0,1,:]==trl_img_reshape[1,:]
trl_img[1,0,:]==trl_img_reshape[1,:]
trl_img[0,2,:]==trl_img_reshape[2,:]
trl_img[0,3,:]==trl_img_reshape[3,:]
trl_img[0,4,:]==trl_img_reshape[4,:]
trl_img[0,5,:]==trl_img_reshape[5,:]
trl_img[0,39,:]==trl_img_reshape[39,:]
tmp_predict=tmp_model.predict(trl_img_reshape)
tmp_predict.shape
trl_img.shape
import matplotlib.pyplot as plt
prd_img_reshape=tmp_predict.reshape(trl_img.shape[0],trl_img[1])
prd_img_reshape=tmp_predict.reshape(trl_img.shape[0],trl_img.shape[1])
prd_img_reshape.shape
plt.imshow(prd_img_reshape)
np.unique(prd_img_reshape)
np.unique(trl_img)
text_imgs[0]
trl_text_imgs[0]
tmp_model.get_params
trn_report=np.load('f1_weighted_SVM_train_report_20191115-202416.npy')
trn_report.item()
trn_report.item().keys()
trn_report.item().get('best_parameters')
trn_report.item().get('best_score')
trn_report.item().get('best_score_')
import pickle
trn_report=pickle.load('SVM_f1_weighted_20191117-010510_test_report')
import pandas as pd
trl_df=pd.read_csv('SVM_f1_weighted_20191117-010510_test_report')
tlr_df
trl_df
pred_img_dict={}
for vals in trl_text_imgs[:5]:
    print(os.path.basename(vals))
    
import os
for vals in trl_text_imgs[:5]:
    print(os.path.basename(vals))
    
for vals in trl_text_imgs:
    tmp_nm=os.path.basename(vals)
    tmp_img=np.load(vals)
    tmp_img_dict={}
    tmp_img_dict['image_orig_shp']=tmp_img.shape
    tmp_img_dict['image_arr_reshp']=tmp_img.reshape(tmp_img.shape[0]*tmp_img.shape[1],tmp_img.shape[2])
    pred_img_dict[tmp_nm]=tmp_img_dict
    
pred_img_dict={}
good_pred_dict={}
for k,v in tmp_img_dict.items():
    tmp_prediction=tmp_model.predict(v['image_arr_reshp'])
    pred_img_dict[k]=tmp_prediction.reshape(v['image_orig_shp'][0],v['image_orig_shp'][1])
    uniq_vals=np.unique(tmp_prediction,return_counts=True)
    if uniq_vals[0].shape[0]>1:
        print(k)
        good_pred_dict[k]=uniq_vals
        
list(tmp_img_dict.keys())[0]
list(tmp_img_dict.keys())[1]
list(tmp_img_dict.keys())[2]
list(tmp_img_dict.keys())[0]
for vals in trl_text_imgs:
    tmp_nm=os.path.basename(vals)
    tmp_img=np.load(vals)
    tmp_img_dict={}
    tmp_img_dict['image_orig_shp']=tmp_img.shape
    tmp_img_dict['image_arr_reshp']=tmp_img.reshape(tmp_img.shape[0]*tmp_img.shape[1],tmp_img.shape[2])
    pred_img_dict[tmp_nm]=tmp_img_dict
    
pred_img_dict.keys()
trn_data_dict={}
pred_img_dict={}
good_pred_dict={}
for vals in trl_text_imgs:
    tmp_nm=os.path.basename(vals)
    tmp_img=np.load(vals)
    tmp_img_dict={}
    tmp_img_dict['image_orig_shp']=tmp_img.shape
    tmp_img_dict['image_arr_reshp']=tmp_img.reshape(tmp_img.shape[0]*tmp_img.shape[1],tmp_img.shape[2])
    trn_data_dict[tmp_nm]=tmp_img_dict
    
pred_img_dict={}
good_pred_dict={}
for k,v in tmp_img_dict.items():
    tmp_prediction=tmp_model.predict(v['image_arr_reshp'])
    pred_img_dict[k]=tmp_prediction.reshape(v['image_orig_shp'][0],v['image_orig_shp'][1])
    uniq_vals=np.unique(tmp_prediction,return_counts=True)
    if uniq_vals[0].shape[0]>1:
        print(k)
        good_pred_dict[k]=uniq_vals
        
pred_img_dict={}
good_pred_dict={}
for k,v in trn_data_dict.items():
    tmp_prediction=tmp_model.predict(v['image_arr_reshp'])
    pred_img_dict[k]=tmp_prediction.reshape(v['image_orig_shp'][0],v['image_orig_shp'][1])
    uniq_vals=np.unique(tmp_prediction,return_counts=True)
    if uniq_vals[0].shape[0]>1:
        print(k)
        good_pred_dict[k]=uniq_vals
        
good_pred_dict
np.unique(tmp_prediction)
v['image_arr_reshp'].shape
tmp_model.get_params
smote_tek_path=r'F:\Biomedical images\Texture_images\texture_imgs_svm_training\texture_imgs_smotetek'
trn_data_smotetek_dict={}
trl_text_imgs_smotetek_lst=list(pathlib.Path(smote_tek_path).rglob('*.npz'))
for vals in trl_text_imgs_smotetek_lst:
    tmp_nm=os.path.basename(vals)
    tmp_img=np.load(vals)
    trn_data_smotetek_dict[tmp_nm]=tmp_img
    
trn_data_smotetek_dict.keys()
trn_data_smotetek_dict['pat_id_8_t1dual_inphase_slice_no_9_256grey_lvl_256x256_W32O1PcdehA0_W16O1PdhA0_W9O1PdcA0_W7O1PdcA0.npz'].keys()
trn_data_smotetek_dict['pat_id_8_t1dual_inphase_slice_no_9_256grey_lvl_256x256_W32O1PcdehA0_W16O1PdhA0_W9O1PdcA0_W7O1PdcA0.npz']['features'].shape
tmp_d_stack=None
num_examples_perc=0.05
list(trn_data_dict.keys())[0]
img_agg_subsamp=None
import random
for k,v in trn_data_dict.items():
    tmp_arr=v['image_arr_reshp']
    tmp_arr_rand_idx=np.random.randint(0, tmp_arr.shape[0], 0.05*tmp_arr.shape[0])
    tmp_arr_sub_samp=tmp_arr[tmp_arr_rand_idx]
    if img_agg_subsamp is None:
        img_agg_subsamp=tmp_arr_sub_samp
    else:
        img_agg_subsamp=np.vstack(img_agg_subsamp,tmp_arr_sub_samp)
        
import random
for k,v in trn_data_dict.items():
    tmp_arr=v['image_arr_reshp']
    tmp_arr_rand_idx=np.random.randint(0, tmp_arr.shape[0], int(0.05*tmp_arr.shape[0]))
    tmp_arr_sub_samp=tmp_arr[tmp_arr_rand_idx]
    if img_agg_subsamp is None:
        img_agg_subsamp=tmp_arr_sub_samp
    else:
        img_agg_subsamp=np.vstack(img_agg_subsamp,tmp_arr_sub_samp)
        
import random
for k,v in trn_data_dict.items():
    tmp_arr=v['image_arr_reshp']
    tmp_arr_rand_idx=np.random.randint(0, tmp_arr.shape[0], int(0.05*tmp_arr.shape[0]))
    tmp_arr_sub_samp=tmp_arr[tmp_arr_rand_idx]
    if img_agg_subsamp is None:
        img_agg_subsamp=tmp_arr_sub_samp
    else:
        img_agg_subsamp=np.vstack((img_agg_subsamp,tmp_arr_sub_samp))
        
img_agg_subsamp.shape
from sklearn import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(0,1)).fit(img_agg_subsamp)
pred_img_dict={}
good_pred_dict={}
for k,v in tmp_img_dict.items():
    tmp_arr=scaling.transform(v['image_arr_reshp'])
    tmp_prediction=tmp_model.predict(tmp_arr)
    pred_img_dict[k]=tmp_prediction.reshape(v['image_orig_shp'][0],v['image_orig_shp'][1])
    uniq_vals=np.unique(tmp_prediction,return_counts=True)
    if uniq_vals[0].shape[0]>1:
        print(k)
        good_pred_dict[k]=uniq_vals
        
pred_img_dict={}
good_pred_dict={}
for k,v in trn_data_dict.items():
    tmp_arr=scaling.transform(v['image_arr_reshp'])
    tmp_prediction=tmp_model.predict(tmp_arr)
    pred_img_dict[k]=tmp_prediction.reshape(v['image_orig_shp'][0],v['image_orig_shp'][1])
    uniq_vals=np.unique(tmp_prediction,return_counts=True)
    if uniq_vals[0].shape[0]>1:
        print(k)
        good_pred_dict[k]=uniq_vals
        
good_pred_dict
plt.imshow(pred_img_dict['pat_id_8_t1dual_inphase_slice_no_9_256grey_lvl_256x256_W32O1PcdehA0_W16O1PdhA0_W9O1PdcA0_W7O1PdcA0.npy'])
import pickle
plt.imshow(pred_img_dict['pat_id_8_t1dual_inphase_slice_no_6_256grey_lvl_256x256_W32O1PcdehA0_W16O1PdhA0_W9O1PdcA0_W7O1PdcA0.npy'])
with open('svm_predicted_images_500_sample') as fb:
    pickle.dump(pred_img_dict)
    
with open('svm_predicted_images_500_sample','wb') as fb:
    pickle.dump(pred_img_dict)
    
with open('svm_predicted_images_500_sample','wb') as fb:
    pickle.dump(pred_img_dict,fb)
    
ls
from sklearn.metrics.pairwise import additive_chi2_kernel
dirs(additive_chi2_kernel)
dir(additive_chi2_kernel)
get(additive_chi2_kernel)
additive_chi2_kernel.__getattr__
additive_chi2_kernel.__getattribute__
additive_chi2_kernel.__getattribute__()
additive_chi2_kernel.__init__
additive_chi2_kernel.__init__()
additive_chi2_kernel?