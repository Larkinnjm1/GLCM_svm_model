# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 23:47:37 2019

@author: niall
"""
import matplotlib.pyplot as plt
import numpy as np
import ipdb
def gen_img_visual(tmp_img,pred,mask,output_path):
    
    cls_dict = {'background':0,'liver':63,'r_kidney':126,'l_kidney':189,'spleen':252}
    classes=['l_kidney','liver','r_kidney','spleen']
    class_values = [cls_dict[clns.lower()] for clns in classes]
    
    mask_logit_idx_slc_unet={'background':(4,4),'l_kidney':(0,0),'r_kidney':(2,2),
                             'liver':(1,1),'spleen':(3,3)}
    try:

        gt_mask=gen_binary_mask(mask,class_values)
        pr_mask=gen_binary_mask(pred,class_values)
    except:
        ipdb.set_trace()
    #Generate visualisation on per slice basis
    visualize(output_path,
        image=tmp_img,
        gt_mask_l_kidney=gt_mask[:,:,mask_logit_idx_slc_unet['l_kidney'][1]],
        pr_mask_l_kidney=pr_mask[:,:,mask_logit_idx_slc_unet['l_kidney'][0]],
        gt_mask_liver=gt_mask[:,:,mask_logit_idx_slc_unet['liver'][1]],
        pr_mask_liver=pr_mask[:,:,mask_logit_idx_slc_unet['liver'][0]],
        gt_mask_r_kidney=gt_mask[:,:,mask_logit_idx_slc_unet['r_kidney'][1]],
        pr_mask_r_kidney=pr_mask[:,:,mask_logit_idx_slc_unet['r_kidney'][0]],
        gt_mask_spleen=gt_mask[:,:,mask_logit_idx_slc_unet['spleen'][1]],
        pr_mask_spleen=pr_mask[:,:,mask_logit_idx_slc_unet['spleen'][0]],
        gt_mask_background=gt_mask[:,:,mask_logit_idx_slc_unet['background'][1]],
        pr_mask_background=pr_mask[:,:,mask_logit_idx_slc_unet['background'][0]],
    )  
    

def visualize(fig_nm=None,figdim=(33,3.1),**images):
    """PLot images in one row."""
    n = len(images)
    print(fig_nm)
    plt.figure(figsize=figdim)
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    
    if fig_nm is not None:
        plt.savefig(fig_nm,dpi=150)
        plt.clf()
    else:
        plt.show()    


def gen_binary_mask(mask:np.ndarray,class_values:list,reord_stack=None)->np.ndarray:
    
    # extract certain classes from mask (e.g. cars)
    masks = [(mask == v) for v in class_values]
    mask = np.stack(masks, axis=-1).astype('float')

    # add background if mask is not binary
    if mask.shape[-1] != 1:
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate((mask, background), axis=-1)
    if reord_stack is None:
        
        return mask
    else:
        return np.transpose(mask,reord_stack)
        
