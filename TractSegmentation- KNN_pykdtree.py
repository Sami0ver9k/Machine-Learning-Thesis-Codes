
"""Nearest Neighbour applied for bundle segmentation  
"""
import numpy as np

from dipy.viz import window, actor
from dipy.tracking.streamline import transform_streamlines
import vtk.util.colors as colors
from dipy.tracking import utils
import time
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.vox2track import streamline_mapping
from pykdtree.kdtree import KDTree
import nibabel as nib


def show_tract(segmented_tract, color):
    """Visualization of the segmented tract.
    """ 
    affine=utils.affine_for_trackvis(voxel_size=np.array([1.25,1.25,1.25]))
    bundle_native = transform_streamlines(segmented_tract, np.linalg.inv(affine))

    renderer = window.Renderer()
    stream_actor2 = actor.line(bundle_native, linewidth=0.1)
    bar = actor.scalar_bar()
    renderer.add(stream_actor2)
    renderer.add(bar)
    window.show(renderer, size=(600, 600), reset_camera=False)          
    """Take a snapshot of the window and save it
    """
    window.record(renderer, out_path='bundle2.1.png', size=(600, 600))


def dsc(estimated_tract, true_tract):
    """Compute the overlap between the segmented tract and ground truth tract
    """
    aff=np.array([[-1.25, 0, 0, 90],[0, 1.25, 0, -126],[0, 0, 1.25, -72],[0, 0, 0, 1]])
    voxel_list_estimated_tract = streamline_mapping(estimated_tract, affine=aff).keys()
    voxel_list_true_tract = streamline_mapping(true_tract, affine=aff).keys()
    TP = len(set(voxel_list_estimated_tract).intersection(set(voxel_list_true_tract)))
    vol_A = len(set(voxel_list_estimated_tract))
    vol_B = len(set(voxel_list_true_tract))
    DSC = 2.0 * float(TP) / float(vol_A + vol_B)
    return DSC   
 
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
    return KDTree(points,leafsize =leafsize)    
    
def kdtree_query(tract,kd_tree):
    """compute 1 NN using kdtree query and return the id of NN
    """
         
    dist_kdtree, ind_kdtree = kd_tree.query(tract, k=1)
    return np.hstack(ind_kdtree) 

def compute_dsc(estimated_tract, filename_true_tract):
    """Comparison between the segmented tract and ground truth tract
    """    
    true_tract=load(filename_true_tract) 
    return dsc(estimated_tract, true_tract)

def segmentation_with_NN(filename_tractogram, filename_example_tract,tractList,no_of_points,leafsize):

    """Nearest Neighbour applied for bundle segmentation 
    """   
    #load tractogram
    print("Loading tractogram: %s" %filename_tractogram)
    tractogram=load(filename_tractogram) 
        
    #load tract
    print("Loading example tract: %s" %filename_example_tract)
    tract=loadSubTract(filename_example_tract,tractList)
    
    t0=time.time()
    #resample whole tractogram
    print("Resampling tractogram........" )
    resample_tractogram=resample(tractogram,no_of_points=no_of_points)
        
    #resample example tract
    print("Resampling example tract.......")
    resample_tract=resample(tract,no_of_points=no_of_points)
    
    #build kdtree
    print("Buildng kdtree")
    kd_tree=build_kdtree (resample_tractogram, leafsize=leafsize)
    
    #kdtree query to retrive the NN id
    query_idx=kdtree_query(resample_tract, kd_tree)
    
    #extract the streamline from tractogram
    estimated_tract=tractogram[query_idx]  

    print("Total amount of time to segment the bundle is %f seconds" % (time.time()-t0))   
    return  estimated_tract

def loadSubTract(subjectList,tractList):
    s=[]
    tractLen=[]
    for i in range(0,len(subjectList)) :
       
        #read tract
        T_filename=subjectList[i]+tractList
        subTract= load(T_filename) 
        tractLen.append(len(subTract))
        s=s+np.array(subTract).tolist()
        
    return np.array(s)

if __name__ == '__main__':
    
    
    target="111312"
    target_brain="full1M_"+target+".trk"
    subjectList =[ "124422"]
    tractList = "_cg.right.trk" 
    ###########Define Filenames#############################################
    filename_tractogram = target_brain
    filename_example_tract = subjectList
    filename_true_tract = target + tractList
    
    # Main parameters:
    no_of_points=12 # number of points for resampling
    leafsize=10    #number of leaf size for kdtree
    
    
    print("Segmenting tract with NN......")   
    estimated_tract= segmentation_with_NN(filename_tractogram, 
                         filename_example_tract,tractList,
                         no_of_points,
                         leafsize)
                                             
      
    print("Computing Dice Similarity Coefficient......")               
    print ("DSC= %f" %compute_dsc(estimated_tract,
                        filename_true_tract))                      
    
    print("Show Segmented tract with NN......")   
    color_positive= colors.green
    color_negative=colors.red
    show_tract(estimated_tract, color_positive)