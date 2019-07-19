# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 04:06:31 2019

@author: u
"""



import numpy as np
from dipy.viz import window, actor
from nibabel import trackvis
from dipy.tracking.streamline import transform_streamlines
import vtk.util.colors as colors
from dipy.viz import fvtk
from dipy.tracking import utils
from sklearn.neighbors import KDTree
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.distances import bundles_distances_mam

from dipy.tracking.vox2track import streamline_mapping
from sklearn import svm
import nibabel as nib
from joblib import Parallel, delayed
import time
from dipy.tracking.utils import length

def comp_dsc(estimated_tract, true_tract):

    aff=np.array([[-1.25, 0, 0, 90],[0, 1.25, 0, -126],[0, 0, 1.25, -72],[0, 0, 0, 1]])
    voxel_list_estimated_tract = streamline_mapping(estimated_tract, affine=aff).keys()
    voxel_list_true_tract = streamline_mapping(true_tract, affine=aff).keys()
    TP = len(set(voxel_list_estimated_tract).intersection(set(voxel_list_true_tract)))
    vol_A = len(set(voxel_list_estimated_tract))
    vol_B = len(set(voxel_list_true_tract))
    DSC = 2.0 * float(TP) / float(vol_A + vol_B)
    return DSC    
 


def show_tract(segmented_tract_positive, color_positive,color_negative,segmented_tract_negative):
   """Visualization of the segmented tract.
   """ 
   ren = fvtk.ren()           
   fvtk.add(ren, fvtk.line(segmented_tract_positive.tolist(),
                           colors=color_positive,
                           linewidth=2,
                           opacity=0.3))
#   fvtk.add(ren, fvtk.line(segmented_tract_negative.tolist(),
#                           colors=color_negative,
#                           linewidth=2,
#                           opacity=0.3))                         
   fvtk.show(ren)
   fvtk.clear(ren)

def load(filename):
    """Load tractogram from TRK file 
    """
    wholeTract= nib.streamlines.load(filename)  
    wholeTract = wholeTract.streamlines
    return  wholeTract 

def resample(streamlines, no_of_points):
    """Resample streamlines using 12 points and also flatten the streamlines
    """
    return np.array([set_number_of_points(s, no_of_points).ravel() for s in streamlines]) 
    
def build_kdtree(points, leafsize):
    """Build kdtree with resample streamlines 
    """
    return KDTree(points,leaf_size =leafsize)    
    
def kdtree_query(tract,kd_tree):
    """compute 1 NN using kdtree query and return the id of NN
    """
         
    dist_kdtree, ind_kdtree = kd_tree.query(tract, k=10)
    return np.hstack(ind_kdtree) 

def bundles_distances_mam_smarter_faster(A, B, n_jobs=-1, chunk_size=100):
    """Parallel version of bundles_distances_mam that also avoids
    computing distances twice.
    """
    lenA = len(A)
    chunks = chunker(A, chunk_size)
    if B is None:
        dm = np.empty((lenA, lenA), dtype=np.float32)
        dm[np.diag_indices(lenA)] = 0.0
        results = Parallel(n_jobs=-1)(delayed(bundles_distances_mam)(ss, A[i*chunk_size+1:]) for i, ss in enumerate(chunks))
        # Fill triu
        for i, res in enumerate(results):
            dm[(i*chunk_size):((i+1)*chunk_size), (i*chunk_size+1):] = res
            
        # Copy triu to trid:
        rows, cols = np.triu_indices(lenA, 1)
        dm[cols, rows] = dm[rows, cols]

    else:
        dm = np.vstack(Parallel(n_jobs=n_jobs)(delayed(bundles_distances_mam)(ss, B) for ss in chunks))

    return dm

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
 
def create_train_data_set(train_subjectList,tract):
    
    T_filename_full_brain="full1M_161731.trk"
    wholeTractogram = load(T_filename_full_brain)
    
    train_data=[]
    for sub in train_subjectList:  
        print (sub)        
        T_filename=sub+tract
        wholeTract = load (T_filename)       
        train_data=np.concatenate((train_data, wholeTract),axis=0) 
        
     
    ###################kdtree################# 
    print ("train data Shape") 
    print (train_data.shape)  
    t0=time.time()    
    resample_tractogram=resample(wholeTractogram,no_of_points=no_of_points)
    resample_tract=resample(train_data,no_of_points=no_of_points)
    
    
    kd_tree=build_kdtree (resample_tractogram, leafsize=leafsize)
    
    #kdtree query to retrive the NN id
    query_idx=kdtree_query(resample_tract, kd_tree)
    
    #extract the streamline from tractogram
    unique_query_idx= np.unique(np.array(query_idx))

    subsample_tract=wholeTractogram[ unique_query_idx]     
    
    wholeTract=np.array(wholeTract)
    x_train = bundles_distances_mam_smarter_faster(train_data, subsample_tract )
    print("Total amount of time tokdtree is %f seconds" % (time.time()-t0))  
    return x_train,subsample_tract,train_data
    
   
      
if __name__ == '__main__':
    
    
    train_subjectList =["124422"]#"192540","117122",  "192540", "106016", "201111","105115","100307","366446"]
                        #"136833","106016","100408","127933"]
    tract = "_af.right.trk"
    no_of_points=12    
    leafsize=10
        
   
    ################################ Train Data######################################

    print ("Preparing Train Data")
    x_train,subsample_tract,train_data= create_train_data_set(train_subjectList,tract)
    print (x_train.shape)
    
    train_lengths = list(length(train_data))
 
    ################labeling#######################    
#    siz=x_train.size
#    y=np.ones(siz)
    
    ###################### Test Data################################
    testTarget="111312"
    testTarget_brain="full1M_"+testTarget+".trk"
    print ("Preparing Test Data")
    t0=time.time()
    t_filename=testTarget_brain #"124422_af.left.trk"
     

    
    test_data=load(t_filename)  
    test_lengths = list(length(test_data))
    x_test =  bundles_distances_mam_smarter_faster(test_data,subsample_tract )   
       
    print ("test data Shape") 
    print (x_test.shape )
    ##########################################
    
    ###########################one class SVM######################
    gamma_value = 0.0001
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=gamma_value)
    clf.fit(x_train)
    ####################################################
    #x_train= np.array(x_train)
    x_pred_train=clf.predict(x_train.tolist())
    n_error_test = x_pred_train[x_pred_train==-1].size
    print('number of error for training =', n_error_test)
    
    x_pred_test=clf.predict(x_test.tolist())
    n_error_test = x_pred_test[x_pred_test==-1].size
    print('number of error for testing=',n_error_test)
    
    
    ###########################visualize tract######################
    test_data=np.array(test_data)
    segmented_tract_positive= test_data[np.where(x_pred_test==1)]
    segmented_tract_negative= test_data[np.where(x_pred_test==-1)]
    print("Total amount of time to compute svm is %f seconds" % (time.time()-t0)) 
    
    print("Show the tract")
    color_positive= colors.green
    color_negative=colors.red
    show_tract(segmented_tract_positive, color_positive,color_negative,segmented_tract_negative) 
    
    ###########################Calculating Dice Similarity Co-efficient########################### 
    trueTract=load(testTarget + tract)  
    dsc=comp_dsc(segmented_tract_positive,trueTract)
    print("Accuracy: ",dsc)
    
    
