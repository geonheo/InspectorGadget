import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from InspectorG.utils import MakeDirectory
import numpy as np
import glob, os
import sys
import cv2


"""
    Built-in policy functions
"""

def ShearX(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v*img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v*img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Rotate(img, v):  # [-30, 30]
    return img.rotate(v)

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Invert(img, _):
    return PIL.ImageOps.invert(img)

#Equalize image histogram
def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)

#Invert all values above threshold
def Solarize(img, v):  # [0, 256]
    return PIL.ImageOps.solarize(img, v)

#Reduce the number of bits per color channel
def Posterize(img, v):  # [4, 8]
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Color(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Color(img).enhance(v)

def Brightness(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Sharpness(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    w, h = img.size
    v = v*img.size[0]
    x0 = np.random.uniform(w-v)
    y0 = np.random.uniform(h-v)
    xy = (x0, y0, x0+v, y0+v)
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def ResizeX(img, v): #[0.8, 1.2] : +-20%
    w, h = img.size
    return img.resize((int(w*v), h))

def ResizeY(img, v): #[0.8, 1.2] : +-20%
    w, h = img.size
    return img.resize((w, int(h*v)))


def PatternAugPolicy(DTYPE, opers, oper_idxs, aug_per_op = 3, p_path = None, aug_dir = None, img_typ = 'png'):
    """
    Policy-based Augmentation
    
    Args:
        DTYPE: Defect Type
        opers: candidate operations (policy)
        oper_idxs: index list of operations
        aug_per_op: the number of policies per image
        p_path: directory of existed pattern
        aug_dir: directory of augmented pattern
        img_typ: filename extension of images
    """

    if p_path == None:
        p_path = './InspectorG/PATTERN/'+ DTYPE
    if aug_dir == None:
        aug_dir = p_path + '-policy/'
    
    MakeDirectory(aug_dir)

    patterns = [f for f in glob.glob(p_path + "/**."+img_typ)]
    transfs = [opers[i] for i in oper_idxs]

    def find_max_num(dirname, pname):
        maxnum = 0
        for tmp in os.listdir(dirname):
            if pname in tmp:
                basenum = int(tmp.split('_')[-1].split('.')[0])
                if basenum > maxnum:
                    maxnum = basenum
        return maxnum


    for p in patterns:
        pname = p.split('/')[-1].split('.')[0]
        max_basenum = find_max_num(aug_dir, pname)
        img = PIL.Image.open(p)

        for i in range(len(transfs)):
            for j in range(aug_per_op):
                img2 = img
                t, min, max = transfs[i]
                v = np.random.rand()*(max-min) + min
                img2 = t(img2, v)
                print(t, v)
                fname=aug_dir+pname+'_rand_'+str(i*aug_per_op+j+max_basenum)+'.'+img_typ
                cv2.imwrite(fname, np.array(img2))
        print("\n")        
