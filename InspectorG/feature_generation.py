from InspectorG.utils import SliceBbox, ImgViewer, SaveDict, LoadDict
import collections as clct
from tqdm import tqdm
import random
import numpy as np
import cv2
import os


class FeatureGenerator:
    def __init__ (self, imgdict, task_name, aug = None, feature_dir = 'InspectorG/DICT'):
        """
        Args:
            imgdict: Image data path dictionary {Product ID : (PATH, Label)}
            task_name: The task name that you want to save
            aug: Augmentation method ex) GAN or policy
            feature_dir: Directory of feature dictionaries
        """
        self.IMGDICT = imgdict
        self.TASK = task_name if aug == None else task_name+'-'+aug
        self.PAT_DICT = self.make_patdict(self.TASK)
        self.FEATURE_DIR = feature_dir
        self.aug = aug
        self.set_featuredict()
        
    def set_featuredict (self):
        """
        Set existed feature dictionaries or make new dictionary.
        """
        if self.TASK+'.featuredict' in os.listdir(self.FEATURE_DIR):
            self.featuredict = LoadDict(self.TASK, 'featuredict')
        else:
            self.featuredict = dict()
       
    def make_patdict (self, dtype):
        """
        Make pattern dictionary using saved pattern images.
        
        Args:
            dtype: Defect type
        """
        root = 'InspectorG/PATTERN/'+dtype
        res = dict()
        for file in os.listdir(root):
            fname = file.split('.')
            if fname[-1] == 'png' or fname[-1] == 'jpg':
                res[fname[0]] = os.path.join(root, file)
        return res
        
    def fg_function (self, fname, pattern):
        """
        Feature Generation Function using cv2.matchTemplate
        
        Args:
            fname: The directory of the image file
            pattern: The directory of the pattern image
        """
        img = cv2.imread(fname)
        pattern_img = cv2.imread(pattern)

        method = eval('cv2.TM_CCOEFF_NORMED')

        result = cv2.matchTemplate(img, pattern_img, method)
        min_val,max_val,min_loc, max_loc = cv2.minMaxLoc(result)
        y, x = np.shape(pattern_img)[:2]
        
        if abs(max_val) > abs(min_val):
            return abs(max_val), (max_loc[0], max_loc[1], max_loc[0]+x, max_loc[1]+y)
        else:
            return abs(min_val), (min_loc[0], min_loc[1], min_loc[0]+x, min_loc[1]+y)
    
    def GenFeature(self, save = True, print_log = False):
        """
        Generate features of the unlabeled images using fg_function().
        
        Args:
            save: If True, the generated features will be saved
        """
        if len(self.PAT_DICT) == 0:
            print("There is no PATTERN")
            return
        feature_dict = clct.defaultdict(dict)
        print('Generate Features ...')
        
        for patname, patdir in tqdm(self.PAT_DICT.items()):    
            
            ex_pat = lambda a : a[list(a.keys())[0]].keys()
            if len(self.featuredict) != 0:
                pat_list = ex_pat(self.featuredict)
            else:
                pat_list = []
            
            if patname not in pat_list:
                for pid, (imgdir, label) in self.IMGDICT.items():
                    res = self.fg_function(imgdir, patdir)
                    feature_dict[pid][patname] = res
            else:
                if print_log:
                    print('%s -> Already Generated' % patname)
        if len(self.featuredict) != 0 and len(dict(feature_dict)) != 0:
            self.featuredict = combine_featuredict(self.featuredict, dict(feature_dict))
        elif len(self.featuredict) == 0:
            self.featuredict = dict(feature_dict)
            
        if save:
            SaveDict(self.featuredict, self.TASK, 'featuredict')

        
    def LoadPatDict (self):
        """
        Load pattern dictionary.
        """
        if self.aug is not None:
            new_patdict = dict()
            for patname, patdir in self.PAT_DICT.items():
                new_patdict[patname+'-'+self.aug] = patdir
            return new_patdict
        else:
            return self.PAT_DICT
    
    def LoadFeatureDict(self, wo_bbox = True):
        """
        Load feature dictionary.
        
        Args:
            wo_bbox: If True, bounding box coordinates are included
        """
        new_featuredict = dict()
        if len(self.featuredict) != 0: 
            for pid, value in self.featuredict.items():
                tmp = value
                if self.aug is not None:
                    tmp = patname_update(tmp, self.aug)
                if wo_bbox:
                    tmp = without_bbox(tmp)
                new_featuredict[pid] = tmp
        return new_featuredict
    
class FeatureManager:
    def __init__(self, task_name, f_org, f_gan = None, f_policy = None):
        """
        Args:
            task_name: The task name that you want to save
            f_org: FeatureGenerator instance with original pattern
            f_gan: FeatureGenerator instance with GAN-based augmented pattern
            f_policy: FeatureGenerator instance with policy-based augmented pattern
        """
        self.TASK = task_name
        self.f_org = f_org
        self.f_gan = f_gan
        self.f_policy = f_policy

        self.check_task_name()
        self.load_featuredicts()
        
    def check_task_name (self):
        """
        Check if the task name is matched with input FeatureGenerator instance.
        """
        try:
            if self.TASK != self.f_org.TASK:
                raise Exception('FeatureE instance is different from Task name')

            elif self.f_gan is not None and self.TASK+'-GAN' != self.f_gan.TASK:
                raise Exception('FeatureE-GAN instance is different from Task name')

            elif self.f_policy is not None and self.TASK+'-policy' != self.f_policy.TASK:
                raise Exception('FeatureE-policy instance is different from Task name')
        except Exception as e:
            print('Error : ', e)
    
    def load_featuredicts (self):
        """
        Load feature dictionary from FeatureGenerator instance.
        """
        def tmp_f(a):
            if a is None:
                return None
            b = a.LoadFeatureDict()
            return b if len(b) != 0 else None
        
        self.pd_org = tmp_f(self.f_org)
        self.pd_gan = tmp_f(self.f_gan)
        self.pd_policy = tmp_f(self.f_policy)
        self.IMGDICT = self.f_org.IMGDICT
    
    def integrate_featuredict (self, mode):
        """
        Integrate feature dictionaries (GAN, policy)
        
        Args:
            mode: experiment mode
        """
        full_dict = dict()
    
        for pid in self.pd_org.keys():
            tmp_org = self.pd_org[pid]

            if mode == 'gan':
                full_dict[pid] = dict(tmp_org, **self.pd_gan[pid])
            if mode == 'policy':
                full_dict[pid] = dict(tmp_org, **self.pd_policy[pid])
            if mode == 'all':
                full_dict[pid] = dict(tmp_org, **self.pd_policy[pid], **self.pd_gan[pid])
            elif mode == 'org':
                full_dict[pid] = tmp_org
        return full_dict
    
    def integrate_patdict (self, mode, dev_dict, pat_num):
        """
        Integrate pattern dictionaries (GAN, policy)
        
        Args:
            mode: experiment mode
            dev_dict: development set dictionary
            pat_num: the number of pattern
        """
        print('== PATTENRNS ==')
        
        org_patdict = filter_patdict(self.f_org.LoadPatDict(), dev_dict)
        print_str = 'ORG : %s' % len(org_patdict)
        if self.pd_gan is not None:
            if mode == 'gan' or mode == 'all':
                aug_patdict_gan = dict_sampling(self.f_gan.LoadPatDict(), pat_num)  
            else:
                aug_patdict_gan = dict()
            print_str += ', GAN : %s' % len(aug_patdict_gan)
        if self.pd_policy is not None:
            if mode == 'policy' or mode == 'all':
                aug_patdict_policy = dict_sampling(self.f_policy.LoadPatDict(), pat_num)  
            else: 
                aug_patdict_policy = dict()
            print_str += ', Policy : %s' % len(aug_patdict_policy)
        print(print_str)
        
        if mode == 'gan':
            aug_patdict = aug_patdict_gan
        elif mode == 'policy':
            aug_patdict = aug_patdict_policy
        elif mode == 'all':
            aug_patdict =dict(aug_patdict_gan, **aug_patdict_policy)
        elif mode == 'org':
            aug_patdict = dict()

        return dict(org_patdict, **aug_patdict)
    
    def MakeMatrix (self, mode, dev_dict, pat_num = None):
        """
        Make feature matrix, which is input of the Labeler, by integrating whole feature dictionaries. 
        
        Args:
            mode: experiment mode
            dev_dict: development set dictionary
            pat_num: the number of pattern
        """
        try:
            if self.pd_gan is None and (mode == 'gan' or mode == 'all'):
                raise Exception('"mode" is not matched with input PrimEs')
                
            if self.pd_policy is None and (mode == 'policy' or mode == 'all'):
                raise Exception('"mode" is not matched with input PrimEs')

        except Exception as e:
            print('Error : ', e)
            
        print('MODE : %s\n' % mode)
        
        i_featuredict = self.integrate_featuredict(mode)
        i_patdict = self.integrate_patdict(mode, dev_dict, pat_num)
        self.i_featuredict = i_featuredict
        
        i_patlist = i_patdict.keys()
        
        print('\nMake Training Data ...')
        X_tr, Y_tr, X_te, Y_te = [], [], [], []
        
        for pid, labels in i_featuredict.items():
            tmp = [labels[pat] for pat in i_patlist]
            if pid in dev_dict:
                X_tr.append(tmp)
                Y_tr.append(self.IMGDICT[pid][1])
            else:
                X_te.append(tmp)
                Y_te.append(self.IMGDICT[pid][1])

        X_tr, Y_tr = np.array(X_tr), np.array(Y_tr)
        X_te, Y_te = np.array(X_te), np.array(Y_te)
        print('Train : %s, %s, Test : %s, %s' % (np.shape(X_tr), np.shape(Y_tr), np.shape(X_te), np.shape(Y_te)))
        return X_tr, Y_tr, X_te, Y_te
        
        
"""
    Simple functions used in this file.
"""
                
def combine_featuredict(p1, p2):
    for key, value in p1.items():
        p1[key] = dict(value, **p2[key])
    return p1

def without_bbox(featuredict):
    return {key : value[0] for key, value in featuredict.items()} 

def patname_update(dic, name):
    return {key+'-'+name : value for key, value in dic.items()} 

def filter_patdict(patdic, devdict):
    tmp = dict()
    for pid, label in devdict.items():
        if label == 1:
            for k, v in patdic.items():
                if pid in k:
                    tmp[k] = v
    return tmp

def dict_sampling(dic, n):
    if n == None:
        n = len(dic.keys())
    n_keys = random.sample(dic.keys(), n)
    return {k : dic[k] for k in n_keys}                
                
                
                
                
                
                
                
                
                
                
        
    
    
    