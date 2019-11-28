
import numpy as np

from glob import glob
import argparse
import os
import json
import ipdb
import pandas as pd
#import progressbar
import pickle as pkl

from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier
import time
#import mahotas as mt
from scipy import stats
import imageio
from haralick_feat_gen import haralick_features
from grey_scale_bkgrnd_foregrnd_seg import img_grey_scale_preprocess
from imblearn.combine import SMOTETomek
from sklearn.svm import LinearSVC, SVC

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
#from pipelinehelper import PipelineHelper
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.kernel_approximation import Nystroem,AdditiveChi2Sampler



def check_args(args):
    print(args.image_train_dir)
    
    if not os.path.exists(args.image_train_dir):
        raise ValueError("Image directory does not exist")

    if not os.path.exists(args.image_test_dir):
        raise ValueError("Label directory does not exist")

    if str(args.classifier).lower() not in ["log_reg","svm_chi","svm_linear","svm_nystrom","rf","gbc"]:
        raise ValueError("Classifier must be either Log reg SVM, RF or GBC")

    #if args.output_model.split('.')[-1] != "p":
     #   raise ValueError("Model extension must be .p")

    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-trn", "--image_train_dir" , help="Path to images/mask folder training directory", required=True)
    parser.add_argument("-tst", "--image_test_dir", help="Path to images/mask folder test directory", required=True)
    parser.add_argument('-txt','--texture_dir',help='Destination path to texture feature files created during analysis',required=True)
    parser.add_argument("-c", "--classifier", help="Classification model to use", required = True)
    parser.add_argument("-o", "--output_model_dir", help="Path to save model. Must end in .p", required = True)
    parser.add_argument('-h_lick_p',"--haralick_params",help="Path to json dictionary of haralick features",required=True)
    parser.add_argument('-model_p',"--model_parameters",help="Path to json model parameters for testing and training",required=False)
    #parser.add_argument('',"",help="",required=True)
    args = parser.parse_args()
    return check_args(args)

def read_data(bs_dir):
    """The purpose of this method is to read in the dataset for analysis"""
    print ('[INFO] Reading image data.')
    
    image_dir=os.path.join(bs_dir,'images')
    
    filelist = glob(os.path.join(image_dir, '*.png'))

    file_img_info={}
    for file in filelist:
        #ipdb.set_trace()
        img=imageio.imread(file)
        f_b_name=os.path.splitext(os.path.basename(file))[0]
        #Getting file labels for analysis
        bs_dir=os.path.dirname(image_dir)
        label=imageio.imread(os.path.join(bs_dir,
                                          'masks',
                                          os.path.basename(file)))
        #Cropped image segmented using method. 
        img_crop,label_crop=img_grey_scale_preprocess(img,label,21,20)
        #Image analysis f basename
        file_img_info[f_b_name]={'image':img_crop,'mask':label_crop}
        
    

    return file_img_info

def subsample(features, labels, low, high, sample_size):

    idx = np.random.randint(low, high, sample_size)

    return features[idx], labels[idx]

def subsample_idx(low, high, sample_size):

    return np.random.randint(low,high,sample_size)

def concat_lst_int(list_int:list,single_chr=False)->str:
    """The purpose of this method is to concatenate lists  of integers into single string of numbers together
    if integer is present it will be converted to a string"""
    #If single character selected single character taken from string to concatenate
    try:
        
        if single_chr:
            tmp_str_concat=''.join(str(x[0]) for x in list_int)
        else:
            tmp_str_concat=''.join(str(x) for x in list_int)
    #If error is type error then it will be converted to a integer value 
    except TypeError as e:
        assert type(list_int)==int,'String int value not present convert window to integer value'
        
        tmp_str_concat=str(list_int)
        
    return tmp_str_concat

def gen_texture_img(img:np.ndarray,
                    haralick_params:dict)->np.ndarray:
    """The purpose of this method is to generate a texture image for a given list of feature parameters and stack them"""
    
    text_rast_img=None
    for param_sets in haralick_params:
        
        #Generating haralick features based on parameter setpoint 
        #Dynamic grey level setting placed into system based on unique grey levels present in the image. 
        texture_img = haralick_features(img,param_sets['WINDOW'],param_sets['OFFSET'],
                                        param_sets['ANGLE'],256, #np.unique(img)[0].shape
                                        param_sets['PROPS'])
        
        if text_rast_img is None:
            text_rast_img=texture_img
        else:
            #Stacking each raster via depth 
            text_rast_img=np.dstack((text_rast_img,texture_img))
    
    return text_rast_img


def gen_text_img_f_name(f_b_name:str,haralick_ftrs_lst:list,single_concat_lst=['props'])->str:
    """The purpose of this method is to generate a texture image file name based on the
    haralick feature parameters specified and the original basename"""
    
    tmp_str_final=''
    #
    for dicts in haralick_ftrs_lst:
        tmp_str_dict=''
        for k,v in dicts.items():
            
            if k.lower() in single_concat_lst:
                
                tmp_str=k[0]+concat_lst_int(v,True)
            else:
                tmp_str=k[0]+concat_lst_int(v,False)
                
            tmp_str_dict=tmp_str_dict+tmp_str
        #Writing dictionary files to file
        tmp_str_final=tmp_str_final+'_'+tmp_str_dict
        
    return f_b_name+tmp_str_final
            
            
    
def create_features(f_b_name:str,
                    img:np.ndarray,
                    label:np.ndarray,
                    haralick_params:list,
                    text_dir:str,
                    model_nm=None,
                    train=True):
    
    #Model name dictates the sampling procedure
    if model_nm.lower() in ["svm_chi","svm_linear","svm_nystrom"]:
        num_examples_perc=0.05
    else:
        num_examples_perc=0.005# number of examples per image to use for training model
    #Determine the number of unique values present in the mask 
    
    #Geneating texture image if filename required. 
    file_nm=gen_text_img_f_name(f_b_name,haralick_params)
    tmp_nm=os.path.join(text_dir,'texture_imgs_raw',file_nm)
    #Generate sub sampling of array using smotetek method. 
    tmp_nm_subsample=os.path.join(text_dir,'texture_imgs_smotetek',file_nm)
    
    if train==True:
        features,label_flat=gen_train_txt_data(img,label,haralick_params,num_examples_perc,tmp_nm,tmp_nm_subsample)

    else:
        features,label_flat=gen_txt_feat(tmp_nm,img,label,haralick_params)
        
    
    return  features,label_flat

def gen_train_txt_data(img,label,haralick_params,num_examples_perc,tmp_nm,tmp_nm_subsample):
    """Purpose of this wrapper function is to provide sub sampled SMOTE data for training. """
    #If file already exists perform subsampling. 
    if os.path.isfile(tmp_nm_subsample+'.npz'):
        
        tmp_file=np.load(tmp_nm_subsample+'.npz')
        
        features_smt=tmp_file['features']
        labels_smt=tmp_file['labels']
              
    else:
        features,label_flat=gen_txt_feat(tmp_nm,img,label,haralick_params)
        features_smt,labels_smt=gen_smote_labls(features,label_flat)
        
    
    features_ss,labels_ss=sub_sample_wrapper(features_smt,
                                             labels_smt,
                                             num_examples_perc)
    
    return features_ss,labels_ss

def gen_txt_feat(tmp_nm,img,label,haralick_params):
    
    #Load texture image if present
    text_rast_img=gen_text_img(tmp_nm,img,haralick_params)
        #Flattening image out into shappe method for analysis
    features = text_rast_img.reshape(text_rast_img.shape[0]*text_rast_img.shape[1],
                                         text_rast_img.shape[2])

    #Class based subsampling required in order to get this system to operate effectively.
    label_flat=label.reshape((label.shape[0]*label.shape[1],))
    
    return features,label_flat

def gen_text_img(tmp_nm,img,haralick_params):
    
    if os.path.isfile(tmp_nm+'.npy'):
            text_rast_img=np.load(tmp_nm+'.npy')
    else:
        text_rast_img=gen_texture_img(img,haralick_params)
        #Including grey level for texture analysis
        text_rast_img=np.dstack((text_rast_img,img))
        
        np.save(tmp_nm,text_rast_img)
       
    return text_rast_img

def sub_sample_wrapper(features_smt,label_smt,num_examples_perc):
    """Wrapper function to perform subsampling of features and labels"""
    #try:
    num_examples=int(features_smt.shape[0]*num_examples_perc)
   # ipdb.set_trace() 
    ss_idx = subsample_idx(0, features_smt.shape[0],num_examples)
    features_ss = features_smt[ss_idx]
    labels_ss = label_smt[ss_idx]
    #except UnboundLocalError as e:
     #   print('Error during sub sampling:',e)
      #  ss_idx = subsample_idx(0, features.shape[0], num_examples)
       # features_ss = features[ss_idx]
        #labels_ss = label_flat[ss_idx]
    
    return features_ss,labels_ss

def gen_smote_labls(features,label_flat):
    
    smt = SMOTETomek(ratio='auto')
    #Performing SMOTE TOMEK over under sampling to boost performance due to class imbalance.
    #Perform only smote if more than one class label is present in the image. 
    if np.unique(label_flat).shape[0]>1:
        
        features_smt, label_smt = smt.fit_sample(features, label_flat)
        #Saving file name  of SMOTEtek array to folder. 
        np.savez(tmp_nm_subsample, features=features_smt, labels=label_smt)
    else:
        np.savez(tmp_nm_subsample, features=features, labels=label_flat)
        
    return features_smt,label_smt

def create_dataset(image_dict:dict,haralick_param:list,text_dir:str,model_nm)->np.ndarray:
    """Wrapper function which takes model input and generated a dataset size dependent on requires dataset size"""
    print ('[INFO] Creating training dataset on %d image(s).' %len(image_dict.keys()))

    X = None
    y = None
    
    for f_b_name,img_arrs in image_dict.items():

        features, labels = create_features(f_b_name,
                                           img_arrs['image'],
                                           img_arrs['mask'],
                                           haralick_param,
                                           text_dir,
                                           model_nm)
        labels=labels[...,np.newaxis]
        if (X is None) and (y is None):
            X=features
            y=labels
        else:
            
            X=np.vstack((X,features))
            y=np.vstack((y,labels))
    
    y=y.squeeze()
 
    print ('[INFO] Feature vector size:', X.shape)

    return X,y#X_train, X_test, y_train, y_test

def train_model(X:np.ndarray, y:np.ndarray, classifier:str):

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

def gen_pipeline(model_nm):
    """Generating pipeline of results based on grid search parameters required. """
    #TODO include argument for paramgrid as json for further use and refactor code into a simplified loop. 
    if model_nm=='log_reg':
        param_grid=[{'ovr__solver':['saga'],'ovr__penalty':['l1', 'l2'],'ovr__C':np.logspace(0, 4, 10),'ovr__multi_class':['ovr','multinomial']},
                   {'ovr__solver':['saga'],'ovr__penalty':['elasticnet'],'ovr__C':np.logspace(0, 4, 10),'ovr__multi_class':['ovr','multinomial'],
                    'ovr__l1_ratio':np.array([0.1,0.3,0.5,0.9])},
                    {'ovr__solver':['sag'],'ovr__penalty':[ 'l2'],'ovr__C':np.logspace(0, 4, 10),'ovr__multi_class':['ovr','multinomial']}]
        OVR_pipe=Pipeline([('ovr',LogisticRegression(random_state=0,max_iter=1000)),]) 
        
    elif model_nm.lower()=='svm_nystrom':
        #
        
        param_grid=[{'nystreum__gamma':[100,10,1,0.1],'nystreum__n_components':[300,60,11],'nystreum__kernel':['rbf'],
                    'ovr__penalty':['l1', 'l2'],'ovr__loss':['hinge', 'modified_huber', 'perceptron']},
            {'nystreum__gamma':[100,10,1,0.1],'nystreum__n_components':[300,60,11],'nystreum__kernel':['sigmoid','polynomial'],
                    'ovr__penalty':['l2'],'ovr__loss':['hinge', 'modified_huber', 'perceptron']}]
        
        OVR_pipe=Pipeline([('nystreum',Nystroem(random_state=1)),
                         ('ovr',SGDClassifier(max_iter=5000, tol=1e-3)),]) #BaggingClassifier(SVC(random_state=0,max_iter=1000),n_estimators=50)
            
    elif model_nm.lower()=='svm_linear':
        param_grid = {'ovr__base_estimator__C': [10, 100, 1000], 'ovr__base_estimator__kernel': ['linear']}
        
        svc_pipe = Pipeline([('svc', SVC()),],verbose=True)
        
        OVR_pipe=Pipeline([('bag', BaggingClassifier(svc_pipe)),],verbose=True) 
        
    elif model_nm.lower()=='svm_chi':
        param_grid=[{'chi_sqr__sample_steps':[1,2,3],
                    'ovr__penalty':['l1', 'l2'],'ovr__loss':['hinge', 'modified_huber', 'perceptron']}]
            
        OVR_pipe=Pipeline([('chi_sqr',AdditiveChi2Sampler()),
                         ('ovr',SGDClassifier(max_iter=5000, tol=1e-3)),]) 
            
    else:
        raise Exception("Grid seach is only possible for SVM and Logistic regression classifiers.")
        
    return OVR_pipe,param_grid

def min_max_scaling(X_train,X_test=None,min_sp=0,max_sp=1,neg_switch=True):
    
    scaling = MinMaxScaler(feature_range=(min_sp,max_sp)).fit(X_train)
    X_train = scaling.transform(X_train)
    if X_test is None:
        pass
    else:
        
        X_test = scaling.transform(X_test)  
        #some values become slightly negative reassign to 0
        if np.sum(np.array(X_test.flatten()) <0, axis=0)<10:
            X_test=np.where(X_test<0,0,X_test)
        else:
            raise ValueError('Too many negative values please review x test versus X_train')
        
    return X_train,X_test

def run_grd_srch(scores,model_nm,X_train,y_train,X_test,y_test,model_dir):
    """Run grid search for analysis"""

    #ipdb.set_trace()    
    #Scaling parameters to optimise grid seach performance. 
    X_train,X_test=min_max_scaling(X_train,X_test)
    
    #Generating pipeline and parameter grid for analysis
    OVR_pipe,param_grid=gen_pipeline(model_nm)
    
    train_results=[]
    
    for score in scores:
        
        #ipdb.set_trace()
        #Perform grid seach across model for analysis
        model_grd=perf_grd_srch(OVR_pipe,param_grid,score,X_train, y_train)
        #Acquiring final results from particular scoring functoin method for analysis
         #Assigning string name for analysis
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_nm=model_nm+'_'+score+'_'+timestr
        prnt_result_scrn(model_grd)
        
        #Generating training and test results for evaluation. 
        tmp_res_dict=gen_train_report(model_grd,model_dir,file_nm)
        gen_test_report(model_grd,y_test,X_test,model_dir,file_nm)
        
        #Append temporary results dictoinary
        train_results.append(tmp_res_dict)
        
    with open(os.path.join(model_dir,file_nm+'_summary_aggregated_training_report'+'.json'),'w') as fb:
        json.dump(train_results,fb)

def prnt_result_scrn(clf):
    #printing results to screen following grid search
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    

def gen_train_report(clf,model_dir,file_nm):
    
    results_dict={'best_parameters':clf.best_params_,
                         'best_score_':clf.best_score_,
                         'cv_results_':clf.cv_results_}
        
    file_nm_train_report=file_nm+'_train_report'
    #Writing best model to file directory for models
    joblib.dump(clf.best_estimator_, os.path.join(model_dir,file_nm+'_best_model'))
    #Appending results to file. 
    
    np.save(os.path.join(model_dir,file_nm_train_report),results_dict)
    return results_dict


def perf_grd_srch(OVR_pipe,param_grid,score,X_train, y_train,
                  weights=np.array([ 0.20937129,  6.25282567, 56.52863436, 55.61599307, 35.46404574])):
    #Performing grid search wrt to training and test data with parameter grid already defined.
    if score.find('weight')!=-1:
        
        clf = GridSearchCV(OVR_pipe, param_grid,cv=3,verbose=10,
                                   scoring=score,n_jobs=-1,sample_weight=weights)
    else:
        clf = GridSearchCV(OVR_pipe, param_grid,cv=3,verbose=10,
                                   scoring=score,n_jobs=-1)
    #Generating grid search rsults for analysis
    print('Grid seach started for:',score)        
    clf.fit(X_train, y_train)
    
    return clf
            

def gen_test_report(clf,y_test,X_test,model_dir,file_nm):
    #Writing test report to file
    y_true, y_pred = y_test, clf.predict(X_test)
    
    file_nm_test_report=file_nm+'_test_report'
    #Generating report on test data for analysis
    test_report_raw=classification_report(y_true, y_pred,output_dict=True)
    test_report_df=pd.DataFrame(test_report_raw).transpose()
    #Writing best model to file directory for models
    print(test_report_raw)
    test_report_df.to_csv(os.path.join(model_dir,file_nm_test_report))

      

def main(image_train_dir :str, image_test_dir:str,
         text_dir:str ,classifier:str,
         output_model_dir:str,haralick_param:dict,svm_hyper_param=None):

    start = time.time()
   
    trn_image_dict = read_data(image_train_dir)
    tst_image_dict = read_data(image_test_dir)
    X_train, y_train = create_dataset(trn_image_dict,haralick_param,text_dir,classifier)
    X_test, y_test= create_dataset(tst_image_dict,haralick_param,text_dir,classifier)
    
    if svm_hyper_param is None:
        scores = ['f1_macro','f1_weighted']#f1macro already completed
        #scores=['r2']#,'explained_variance_score','neg_mean_absolute_error','neg_mean_squared_error']
        run_grd_srch(scores,classifier,
                     X_train,y_train,
                     X_test,y_test,
                     output_model_dir)

    #If model is runnning perform grid search where appropriate  
    else:
        assert svm_hyper_param is not None,'No hyper parameter present you cannot train'
        model = train_model(X_train, y_train, classifier,svm_hyper_param)
        test_model(X_test, y_test, model)
        pkl.dump(model, open(output_model, "wb"))
    print ('Processing time:',time.time()-start)

if __name__ == "__main__":
    args = parse_args()
    image_train_dir = args.image_train_dir
    image_test_dir = args.image_test_dir
    text_dir=args.texture_dir
    #ipdb.set_trace()
    classifier = args.classifier
    output_model_dir = args.output_model_dir
    
    with open(args.haralick_params,'r') as file_read:
        h_lick_param=json.load(file_read)   
        
    if args.model_parameters is not None:
        with open(args.model_parameters,'r') as file_read:
            model_param=json.load(file_read)
    else:
        model_param=None
        
    
    main(image_train_dir, image_test_dir,text_dir,classifier, output_model_dir,
         h_lick_param,model_param)
