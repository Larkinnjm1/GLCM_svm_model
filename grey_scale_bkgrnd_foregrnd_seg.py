# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 21:41:42 2019

@author: aczd087
"""
import cv2
import numpy as np


def gen_glob_threshold_img(trl_img1,thresh_sp=0):
    """The purpose of this method is to create a rough binary thresholded image"""
    
    ret, thresh = cv2.threshold(trl_img1, thresh_sp, 255, 0)
    
    return thresh

def gen_larget_contr(thres_img):
    """the purpose of this method is to find the thresholded image for analysis"""
    img,contours, hierarchy = cv2.findContours(thres_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #find the largest contour in the image for analysis and extraction., 
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    
    return biggest_contour

def blur_img(img,blur_kern_s_sp=21):
    """"""
    return cv2.GaussianBlur(img,(blur_kern_s_sp,blur_kern_s_sp),0)

def get_filt_img_bbox(img,mask,largest_cntr):
    """the purpose of this method is to find the bounding box around a specific contour"""
    x,y,w,h = cv2.boundingRect(largest_cntr)
    
    img_filt=img[y:y+h,x:x+w]
    mask_filt=mask[y:y+h,x:x+w]
    
    return (img_filt,mask_filt,(x,y,w,h))

def all_orgs_within_filt_img(old_mask,filt_mask):
    """the purpose of this method is to confirm that all organs are present in the new mask"""
    #Getting new pixel count
    mask_orig_pixel_cnt=np.unique(old_mask,return_counts=True)
    mask_filt_pixel_cnt=np.unique(filt_mask,return_counts=True)
    
    #Get non zero organs only
    idx_nz_orig=np.nonzero(mask_orig_pixel_cnt[0])
    idx_nz_filt=np.nonzero(mask_filt_pixel_cnt[0])
    #Confirming all non background organs match or not. 
    mask_orig_pixel_cnt_non_bkgrnd=mask_orig_pixel_cnt[1][idx_nz_orig]
    mask_filt_pixel_cnt_non_bkgrnd=mask_filt_pixel_cnt[1][idx_nz_filt]
    mask_filt_orgs_non_bkgrnd=mask_filt_pixel_cnt[0][idx_nz_filt]
    #If all organs are captured great then return simple true value if not then return the percentage of the missing values. 
    if all(mask_orig_pixel_cnt[0]==mask_filt_pixel_cnt[0])&all(mask_orig_pixel_cnt_non_bkgrnd==mask_filt_pixel_cnt_non_bkgrnd):
        return (True,'N/A','N/A')
    else:
        orgs_not_cnt=mask_filt_pixel_cnt_non_bkgrnd[mask_orig_pixel_cnt_non_bkgrnd!=mask_filt_pixel_cnt_non_bkgrnd]
        orgs_not_type=mask_filt_orgs_non_bkgrnd[mask_filt_orgs_non_bkgrnd!=mask_filt_pixel_cnt_non_bkgrnd]
        return (False,orgs_not_type,orgs_not_cnt)

    
def img_grey_scale_preprocess(img,mask,blur_sp,thresh_sp):
    """the purpose of this method is to get binary mask that concatenate with texture images for analysis
    to minimise the interference from different organs. """
    #Blurring the original image
    tmp_blur_img=blur_img(img,blur_sp)
    #thresholding image based on blurring. 
    tmp_thresh_img=gen_glob_threshold_img(tmp_blur_img,thresh_sp)
    #Generating contour around thresholded image
    contour_polygon=gen_larget_contr(tmp_thresh_img)
    #Filter mask and img based on contour
    filt_img,filt_mask,bbox_coord=get_filt_img_bbox(img,mask,contour_polygon)
    x,y,w,h=bbox_coord

    #confirming all organs are within bounding box for analysis.
    org_chk_bool,orgs_not_type,orgs_not_cnt=all_orgs_within_filt_img(mask,filt_mask)

    #Placing mask of contour into empty array
    if org_chk_bool==True:
        
        return(img[y:y+h,x:x+w],mask[y:y+h,x:x+w])
        
    else:
        print('Not all organs captured')
        print('organs missing:',orgs_not_type)
        print('organs missing counts:',orgs_not_cnt)
        return(img,mask)
    
