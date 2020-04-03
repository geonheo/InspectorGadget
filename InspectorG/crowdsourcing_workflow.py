from InspectorG.utils import ImgViewer, LoadDict, MakeDirectory
from collections import namedtuple
import numpy as np
import copy
import cv2
import os

PAT_ROOT = 'sample_patterns/'
extract_fname = lambda f : f.split('/')[-1].split('.')[0]
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def ExtractCandidate(org_fname, img_list, margin = 10, show = True):
    """
    Args:
        org_fname: original image file name
        img_list: list of marked images
        margin: margin on the edge of bounding box
        show: If True, show candidate patterns
    """
    org = cv2.imread(org_fname)
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    img = copy.deepcopy(org)
    group = []
    for fname in img_list:
        candidate = cv2.imread(fname)
        height, width = candidate.shape[:2]
        
        hsv = cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 200, 200])
        upper_red = np.array([200, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        tmp = np.where(mask>0)
        max_y, min_y, max_x, min_x = max(tmp[0]), min(tmp[0]), max(tmp[1]), min(tmp[1])
        
        min_x, max_x = max(0, min_x-margin), min(width, max_x+margin)
        min_y, max_y = max(0, min_y-margin), min(height, max_y+margin)
        
        group.append(Rectangle(min_x, min_y, max_x, max_y))
        if show:
            img = cv2.rectangle(img, (max_x, max_y), (min_x, min_y), (255,0,0), 3)
    if show:
        ImgViewer(img)
    return group

def CombinePatterns(org_fname, group, save = False, show = True, defect_label = 'defect'):
    """
    Args:
        org_fname: original image file name
        group: output of ExtractCandidate()
        save: If True, save combined patterns from images
        show: If True, show combined patterns and outliers
        defect_label: The type of defect
    """
    merge_result, outlier = merge(group)
    org = cv2.imread(org_fname)
    img = copy.deepcopy(org)

    for m in merge_result:
        if show:
            img = cv2.rectangle(img, (m.xmax, m.ymax), (m.xmin, m.ymin), (0,255,0), 3)
        if save:
            fname = extract_fname(org_fname)
            crop_and_save(org, m, 'avg', fname, defect_label)

    for o in outlier:
        if show:
            img = cv2.rectangle(img, (o.xmax, o.ymax), (o.xmin, o.ymin), (0, 0, 255), 3)
            
    if show:        
        ImgViewer(img)
    return merge_result, outlier

def PeerReview(org_fname, outlier, save_outlier = False, defect_labels = ['defective']):
    """
    Args:
        org_fname: original image file name
        outlier: output of CombinePatterns()
        save_outlier: If True, save true outliers after peer review
        defect_labels: list of defect types (for multi-class)
    """
    if len(defect_labels) == 1:
        print('Binary-class')
    else:
        print('Multi-class')
    org = cv2.imread(org_fname)
    
    defect_labels = ['No defect'] + defect_labels
    instruction = '   '.join([f'{d} : {i}' for i, d in enumerate(defect_labels)])
    
    for o in outlier:
        img = copy.deepcopy(org)
        img = cv2.rectangle(img, (o.xmax, o.ymax), (o.xmin, o.ymin), (0, 0, 255), 3)
        ImgViewer(img)
        print('Type an integer label corresponding to defect type.\n')
        print(instruction)
        ans = int(input())
        
        fname = extract_fname(org_fname)
        if ans != 0:    
            crop_and_save(org, o, 'peer', fname, defect_labels[ans])
        elif save_outlier:
            crop_and_save(org, o, 'out', fname, defect_labels[ans])   
            
            
def area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0
    
def crop_and_save(org, m, ptype, fname, label):
    crop = org[m.ymin:m.ymax, m.xmin:m.xmax]
    max_count = -1
    for pat in os.listdir(PAT_ROOT):
        tmp = extract_fname(pat).split('_')
        if tmp[0] == fname and tmp[1] == ptype:
            max_count = max(int(tmp[2]), max_count)
    MakeDirectory(PAT_ROOT)
    MakeDirectory(os.path.join(PAT_ROOT, label))
    cv2.imwrite(f'{PAT_ROOT}/{label}/{fname}_{ptype}_{max_count+1}.png', crop)
    print(f'Saved as {PAT_ROOT}/{label}/{fname}_{ptype}_{max_count+1}.png')

def merge(group):
    result = []
    for i in range(len(group)):
        for j in range(i+1,len(group)):
            if area(group[i], group[j]) > 0:
                
                for r in result:
                    if len(set([i,j]).intersection(r)) > 0:
                        r.update([i,j])
                    else:
                        result.append(set([i, j]))
                
                if len(result) == 0:
                    result.append(set([i, j]))
                else:
                    for ii in range(len(result)):
                        for jj in range(ii+1, len(result)):
                            if len(result[ii].intersection(result[jj])) > 0:
                                result[ii].update(result[jj])
                                del result[jj]
    
    outlier = []
    tmp = set()
    for r in result:
        tmp.update(r)
        
    for i in range(len(group)):
        if i not in tmp:
            outlier.append(group[i])
    
    for j, mer in enumerate(result):
        avg = np.zeros(4)
        for i in mer:
            m = group[i]
            avg += np.array([m.xmax, m.ymax, m.xmin, m.ymin])/len(mer)
        avg = avg.astype(int)
        max_x, max_y, min_x, min_y = avg[0], avg[1], avg[2], avg[3]
        result[j] = Rectangle(min_x, min_y, max_x, max_y)

    return result, outlier