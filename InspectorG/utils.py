from matplotlib import pyplot as plt
import numpy as np
import pickle
import copy
import cv2
import os


def LoadDict (dic_name, dic_type = 'dict'):
    """ 
    Load dictionary
    
    Args:
        dic_name: The name of the dictionary
        dic_type: The type of dictionary
    """
    with open('InspectorG/DICT/' + dic_name + '.' + dic_type, 'rb') as file:
        dic = pickle.load(file)
    return dic
    
def SaveDict (dic, dic_name, dic_type = 'dict'):
    """ 
    Save dictionary
    
    Args:
        dic: The dictionary that you want to save
        dic_name: The name of the dictionary
        dic_type: The type of dictionary
    """
    with open('InspectorG/DICT/' + dic_name + '.' + dic_type, 'wb') as file:
        pickle.dump(dic, file)
    return 0

def SliceBbox(img, bbx):
    """ 
    Extract bounding box from original image
    
    Args:
        img: Image array
        bbx: bounding box coordinates
    """
    (max_x, max_y, min_x, min_y) = bbx
    return copy.deepcopy(img[min_y:max_y, min_x:max_x])

def ImgViewer(img):
    """ 
    Show image using matplotlib
    
    Args:
        img: Image array or image path
    """
    if type(img) == str:
        img = cv2.imread(img)
    fig, ax = plt.subplots(figsize=(18, 2))
    ax.imshow(img, interpolation='nearest')
    plt.show()
    plt.tight_layout()
    
def ViewOrgImage(imgdict, pid):
    """ 
    Show image corresponding to pid
    
    Args:
        imgdict: Image data path dictionary {Product ID : (PATH, Label)}
        pid: id for each image.
    """
    ImgViewer(imgdict[pid][0])
    
def MakeDirectory(target_dir):
    """ 
    Make directory on target_dir
    
    Args:
        target_dir: directory that you want to make
    """
    directory = target_dir#os.path.join(target_dir, dirname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
