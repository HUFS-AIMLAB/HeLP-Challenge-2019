from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
import torch
import nibabel as nib
import os
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
import cv2
import skimage.morphology as mp
import scipy.ndimage.morphology as sm


print('starting inference.py')

'''
Load preprocessed dataset
--------------------------------
'''

path = Path('/data/volume')
path.ls()

path_lbl = path/'lbl'
path_img = path/'img'

## Data
fnames = get_image_files(path_img)

lbl_names = get_image_files(path_lbl)
img_f = fnames[4]
img = open_image(img_f)

get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'
mask = open_mask(get_y_fn(img_f))
size = np.array(mask.shape[1:])

codes = np.array(['Void', 'HT', 'PT', 'HP'], dtype=str)

# the max size of bs depends on the available GPU RAM
free = gpu_mem_get_free_no_cache()
if free > 8200: bs=8
else:           bs=8

src = (SegmentationItemList.from_folder(path_img)
       .split_by_rand_pct(valid_pct=0.2)
       .label_from_func(get_y_fn, classes=codes))

data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_wbct(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

metrics=acc_wbct
wd=1e-2
# load pretrained ResNet encoder model
def my_resnet(pretrained=True, progress=True, **kwargs):
  m = models.resnet34(pretrained=False, progress=True, **kwargs)
  m.load_state_dict(torch.load("./resnet34-333f7ec4.pth"))
  return m
learn = unet_learner(data, my_resnet, metrics=metrics, wd=wd)

'''
Load Trained model
--------------------------------
'''
learned = learn.load('/data/volume/model/trained')


'''
Test data preprocessing
--------------------------------
'''
test_path = '/data/test'
test_ls=os.listdir(test_path)
test_img = [file for file in test_ls if file.endswith((".img"))]
test_paths= [os.path.join(test_path, ii) for ii in test_img]

target_pth = '/data/output'

# windowing image
def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image

# remove CT plate
def remove_pan(img):
    imgmin = abs(np.min(img))
    img = img + abs(np.min(img))
    imgmax = np.max(img)
    img = img/np.max(img)
    img = (img*255).round().astype(np.uint8)
    th,img1 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    disk5 = mp.disk(5)
    disk2 = mp.disk(2)
    img2 = mp.binary_opening(img1,disk5)
    img2 = mp.binary_dilation(img2,disk2)
    img2 = sm.binary_fill_holes(img2)
    img2 = (img2*1).astype('uint8')
    masked = cv2.bitwise_or(img,img,mask=img2)
    masked = (((masked/255)*imgmax) - imgmin).astype(np.int16)
    return masked

# convert2array
def cv_arr(filepath):
    img_nifty = nib.load(filepath)
    img_array = img_nifty.get_data()
    return img_array

# flip&rotate
def flip_rot(arr):
    return np.fliplr(np.rot90(arr))

for path in test_paths:
    pid = path.split('/')[-1]
    pnum = pid[:-4]
    img3 = cv_arr(path)
    x, y, z = img3.shape
    img = np.zeros((x, y, z))
    for i in range(z):
        img[:, :, i] = remove_pan(img3[:, :, i])
    # Hemothorax setting
    img_arr = window_image(img, 50, 550)
    img_arr = img_arr + abs(img_arr.min())
    img_arr = img_arr.astype(np.float16)
    img_arr = img_arr / (img_arr.max())
    # Pneumothorax setting
    img_arr1 = window_image(img, -500, 1500)  
    img_arr1 = img_arr1 + abs(img_arr1.min())
    img_arr1 = img_arr1.astype(np.float16)
    img_arr1 = img_arr1 / (img_arr1.max())
    # Hemoperi setting
    img_arr2 = window_image(img, 50, 350)  
    img_arr2 = img_arr2 + abs(img_arr2.min())
    img_arr2 = img_arr2.astype(np.float16)
    img_arr2 = img_arr2 / (img_arr2.max())

    csv1= np.zeros((1,3))
    csv2= np.zeros(((img_arr.shape[2]),3))
    
    test_img = []

    for i in range (img_arr.shape[2]):
        limg=np.zeros((img_arr.shape[1],img_arr.shape[0],3),dtype=np.float16)
        limg[:, :, 0] = flip_rot(img_arr[:, :, i])
        limg[:, :, 1] = flip_rot(img_arr1[:, :, i])
        limg[:, :, 2] = flip_rot(img_arr2[:, :, i])
        test_img.append(limg)


    np_whole=[]    
    for pn in range(len(test_img)):
        
        I = test_img[pn]
        I = I/(I.max()/255.0)
        I = Image(pil2tensor(I, dtype = np.float32).div_(255))

        out1, out2, out3 = learned.predict(I)

        # convert2numpy
        out1=np.flip(out1.data.numpy()[0, :, :])

        # set threshold value 0.25 for "per lesion" & "per slice" task
        o1 = out3.numpy()[1, :, :]
        o1[o1<=0.25] = 0.
        o2 = out3.numpy()[2, :, :]
        o2[o2<=0.25] = 0.
        o3 = out3.numpy()[3, :, :]
        o3[o3<=0.25] = 0.
        
        # number of nonzero pixels
        A=np.count_nonzero(o1)
        B=np.count_nonzero(o2)
        C=np.count_nonzero(o3)

        # set threshold value 0.45 for detection task
        o1_1 = o1.copy()
        o1_1[o1_1<=0.45] = 0.
        o1_2 = o2.copy()
        o1_2[o1_2<=0.45] = 0.
        o1_3 = o3.copy()
        o1_3[o1_3<=0.45] = 0.

        o1_1=np.where(o1_1 !=0,1,o1_1)
        o1_2=np.where(o1_2 !=0,1,o1_2)
        o1_3=np.where(o1_3 !=0,1,o1_3)

        # remove duplicated pixel between labels
        o1_1=np.where(((o1_1!=0)&(o1_1==o1_2)&(o1<o2)),0,o1_1)
        o1_1=np.where(((o1_1!=0)&(o1_1==o1_3)&(o1<o3)),0,o1_1)

        o1_2=np.where(((o1_2!=0)&(o1_2==o1_1)&(o2<o1)),0,o1_2)
        o1_2=np.where(((o1_2!=0)&(o1_2==o1_3)&(o2<o3)),0,o1_2)

        o1_3=np.where(((o1_3!=0)&(o1_3==o1_1)&(o3<o1)),0,o1_3)
        o1_3=np.where(((o1_3!=0)&(o1_3==o1_2)&(o3<o2)),0,o1_3)

        # convert pt,hp label to 2,3
        o1_2=np.where(o1_2 !=0,2,o1_2) 
        o1_3=np.where(o1_3 !=0,3,o1_3)

        # save final 2D slice
        D= o1_1+o1_2+o1_3
        np_whole.append(np.flip(D))

        # csv2 for per slice task
        if A != 0.:
            csv2[pn][0]=np.sum(o1)/A
        if B != 0.:
            csv2[pn][1]=np.sum(o2)/B
        if C != 0.:
            csv2[pn][2]=np.sum(o3)/C
    # save 3D results
    nparr_3 = np.transpose(np.asarray(np_whole).astype(np.float32), (2,1,0))
    probpth_3 = target_pth+'/%s.img'%str(pnum)
    nii_3 = nib.Nifti1Image(nparr_3, affine = np.eye(4))
    nib.save(nii_3, probpth_3)

    # csv1 for per patient task
    if np.count_nonzero(csv2[:, 0]) > 0:
        csv1[0, 0] = np.sum(csv2[:, 0]) / np.count_nonzero(csv2[:, 0]) + np.count_nonzero(csv2[:, 0]) / len(test_img)
        if csv1[0, 0] > 1: csv1[0, 0] = 1
    if np.count_nonzero(csv2[:, 1]) > 0:
        csv1[0, 1] = np.sum(csv2[:, 1]) / np.count_nonzero(csv2[:, 1]) + np.count_nonzero(csv2[:, 1]) / len(test_img)
        if csv1[0, 1] > 1: csv1[0, 1] = 1
    if np.count_nonzero(csv2[:, 2]) > 0:
        csv1[0, 2] = np.sum(csv2[:, 2]) / np.count_nonzero(csv2[:, 2]) + np.count_nonzero(csv2[:, 2]) / len(test_img)
        if csv1[0, 2] > 1: csv1[0, 2] = 1

    with open('/data/output/{}.csv'.format(pnum), 'w', encoding='utf-8', newline='') as file :
        wr = csv.writer(file)
        wr.writerow(['ID',pnum,''])
        wr.writerow(['Hemothorax',csv1[0,0],''])
        wr.writerow(['Pneumothorax',csv1[0,1],''])
        wr.writerow(['Hemoperitoneum',csv1[0,2],''])
        wr.writerow([])
        for a in range (img_arr.shape[2]):
            wr.writerow([csv2[a][0],csv2[a][1],csv2[a][2]])