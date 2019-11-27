import cv2
import numpy as np
import pylab as plt
from glob import glob
import argparse
import os
import pickle as pkl
import train
import math
from joblib import dump, load
import pathlib
from sklearn.preprocessing import MinMaxScaler
from train import create_features

def check_args(args):

    if not os.path.exists(args.image_dir):
        raise ValueError("Image directory does not exist")

    if not os.path.exists(args.output_dir):
        raise ValueError("Output directory does not exist")

    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-txt", "--txt_image_dir" , help="Path to images", required=False)
    parser.add_argument("-i", "--image_dir" , help="Path to images", required=False)
    parser.add_argument("-m", "--model_path", help="Path to .p model", required=True)
    parser.add_argument("-o", "--output_dir", help="Path to output directory", required = True)
    args = parser.parse_args()
    return check_args(args)

def create_features(f_nm,img,args):

    img_blnk = np.zeros(img.shape)

    features, _ = create_features(f_nm,img,img_blnk,args.txt_image_dir, label=None, train=False)

    return features

def compute_prediction(img, model):
    
    img_reshp=img.reshape((img.shape[0]*image.shape[1],image.shape[2]))
    
    

    img = cv2.copyMakeBorder(img, top=border, bottom=border, \
                                  left=border, right=border, \
                                  borderType = cv2.BORDER_CONSTANT, \
                                  value=[0, 0, 0])

    features = create_features(img)
    predictions = model.predict(features.reshape(-1, features.shape[1]))
    pred_size = int(math.sqrt(features.shape[0]))
    inference_img = predictions.reshape(pred_size, pred_size)

    return inference_img

def infer_images(image_dir, model_path, output_dir):

    txt_filelist = list(pathlib.Path(image_dir).rglob('*.npy'))

    print ('[INFO] Running inference on %s test images' %len(filelist))

    model = load(model_path)

    for file in filelist:
        print ('[INFO] Processing images:', os.path.basename(file))
        inference_img = compute_prediction(cv2.imread(file, 1), model)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(file)), inference_img)

def main(image_dir, model_path, output_dir):

    infer_images(image_dir, model_path, output_dir)

if __name__ == '__main__':
    args = parse_args()
    image_dir = args.image_dir
    model_path = args.model_path
    output_dir = args.output_dir
    main(image_dir, model_path, output_dir)
