# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:16:23 2019

@author: niall

"""
import ipdb
import os
import argparse
import pandas as pd
import numpy as np
from train import read_data,create_dataset
from plot_learning_curve import plot_learning_curve
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import glob
import json

def arg_parser():
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-rw_r_dir','--raw_results_dir',
                        help='source results directory where raw results are read in for analysis',
                        required=True)
    parser.add_argument('-dst_dir','--dest_dir',
                        help='destination directory to write output dataframes and plots from analysis',
                        required=True)
    parser.add_argument('-trn_data_dir','--train_data_dir',
                        help='source directory where original raw data is read in for additional training curve plot analysis',
                        required=True)
    parser.add_argument('-tst_data_dir','--test_data_dir',
                        help='source directory where raw test data is read in for additional curve plotting analysis',
                        required=True)
    
    parser.add_argument('-txt_params','--haralick_txt_params',
                        help='source directory to original json of texture features',
                        required=True)

    parser.add_argument('-t','--text_dir',
                        help='Destination path to texture feature files created during analysis',
                        required=True)
    
    return parser.parse_args()
    
def extract_model_type(file_nm_lst:list)->list:
    """The purpose of this method is to take a list of files and return 
    a list of dictionary of the type of model they are and the scoring function to generate them"""
    
    list_return=[]
    #ipdb.set_trace()
    for file in file_nm_lst:
        
        key_words={'score':['f1','jaccard'],
               'score_method':['weighted','macro'],
               'model_type':['svm','log_reg']}
        
        file_bs_nm=os.path.splitext(os.path.basename(file))[0]
        key_words_tmp={k:[x for x in v if file_bs_nm.lower().find(x)!=-1][0] for k,v in key_words.items()}
        key_words_tmp['path']=file
        list_return.append(key_words_tmp)
        
    return list_return
        
def gen_file_lst(results_dir):
    """Generating file list for analysis """
    root_fl=[]
    for root,subdir,files in os.walk(results_dir):
        
        if len(files)>0:
            file_lst=glob.glob(os.path.join(root,'*.npy'))
            root_fl=root_fl+file_lst
    
    return root_fl

def main(args):
    
    #Getting all training reports for analysis and creating json dictionary of information on file. 
    train_reports=gen_file_lst(args.raw_results_dir)
    train_report_detail=extract_model_type(train_reports)
    
    with open(args.haralick_txt_params,'r') as fb:
        haralick_params=json.load(fb)
    #
    trn_image_dict = read_data(args.train_data_dir)
    tst_image_dict = read_data(args.test_data_dir)
    
    #Iterating through reports for analysis
    for data_combos in train_report_detail:
        #Generate training numpy arrays for analysis
        X_train, y_train = create_dataset(trn_image_dict,haralick_params,args.text_dir,data_combos['model_type'])
        X_test, y_test= create_dataset(tst_image_dict,haralick_params,args.text_dir,data_combos['model_type'])
            
        scaling = MinMaxScaler(feature_range=(0,1)).fit(X_train)
        X_train = scaling.transform(X_train)
        X_test = scaling.transform(X_test)
 
        #load data for analysis into dataframe
        tmp_arr_dict=np.load(data_combos['path'],allow_pickle=True)
        tmp_arr_df=tmp_arr_dict.item().get('cv_results_')
        tmp_arr_df=pd.DataFrame.from_dict(tmp_arr_df)
        tmp_arr_df['params'].apply(pd.Series)
        
        #Perform analysis for generating 
        tmp_arr_df.sort_values('rank_test_score',ascending=True,inplace=True)
        trl_arr_df_params_lst=tmp_arr_df['params'][:5].tolist()
        #Restructure file name for analysis
        model_params_reformat=reformat_model_params(trl_arr_df_params_lst)
        #ipdb.set_trace()
        #Taking the top 5 performers forward for running analysis with training and testing curves. 
        for vals in model_params_reformat:
            #Generating detailed tile for model performance.
            title2='_'.join(['_'.join((k,str(v))) for k,v in vals.items()])
            title1='_'.join([v for k,v in data_combos.items() if k!='path'])
            title=title1+'_'+title2
            
            tmp_estimator=gen_estimator(data_combos['model_type'],vals)
            
            tmp_fig=plot_learning_curve(tmp_estimator, title, X_train, y_train,
                                        cv=3,n_jobs=-1)
            #Save figure for analysis
            dst_dir_f=os.path.join(args.dest_dir,title+'.jpeg')
            tmp_fig.savefig(dst_dir_f)

def gen_estimator(model_type:str,model_params:dict)-> sklearn.base.BaseEstimator:
    """The purpose of this method is to generate model parameters for analyss """
    #ipdb.set_trace()
    if model_type.lower()=='svm_sgd':
        
        OVR_pipe=Pipeline([('nystreum',Nystroem(random_state=1)),
                         ('ovr',SGDClassifier(max_iter=5000, tol=1e-3)),])
        
        return OVR_pipe.set_params(**model_params)
    
    elif model_type.lower()=='svm_linear':
        return BaggingClassifier(base_estimator=SVC(random_state=0,max_iter=5000,**model_params),n_estimators=50) 
    elif model_type.lower()=='log_reg':
        return LogisticRegression(max_iter=5000,**model_params)
    else:
        raise ValueError
    
    
def reformat_model_params(trl_arr_df_params_lst:list)->list:
    """The purpose of this method is to reformat the keys of the model parameters"""
    reformat_lst=[]
    
    for params in trl_arr_df_params_lst:
        tmp_dict={}
        for k,v in params.items():
            splt_lst=k.split('__')[-1]
            if type(v) is np.float64:
                v=round(v,2)
            tmp_dict[splt_lst]=v
        #ipdb.set_trace()
        reformat_lst.append(tmp_dict)
    
    return reformat_lst
    
    
if __name__=='__main__':
    
    args=arg_parser()
    
    main(args)
