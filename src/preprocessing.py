# -*- coding: utf-8 -*-
import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mat
import random
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from skimage.util import random_noise

base_path = '/data/train'
img_path='/data/train/img'
gt_path='/data/train/groundtruth'

training_i=os.listdir(img_path) # original img
training_g=os.listdir(gt_path) # original gt
training_img = [file for file in training_i if file.endswith(("_P.img","_A.img",".img"))]# only .img file
training_gt = [file for file in training_g if file.endswith(("_P.img","_A.img",".img"))]# only .img file

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

# make noise
def noise(im):
  img_r = random_noise(im)
  return img_r

# add tranformed data
def augmentation(lb,num):
  for pid in lb:
    img=open_image(os.path.join(img_png, '{}.png'.format(pid)))
    mask=open_image(os.path.join(gt_png, '{}_P.png'.format(pid)))
    tfms = get_transforms(do_flip=True, max_rotate=25)
    # add noise in last phase
    for i in range(num):
      if i==num-1: 
        A=img.data.numpy()
        A=np.transpose(A,(1,2,0))
        A = A.astype(np.float16)
        A = noise(A)
        B=mask.data.numpy()
        B=np.transpose(B,(1,2,0))
        B = B.astype(np.uint8)
        plt.imsave(os.path.join(img_png, '{}_{:03d}.png'.format(pid,i)),A)
        cv2.imwrite(os.path.join(gt_png, '{}_{:03d}_P.png'.format(pid,i)),B[:,:,0])
      else: 
        A=img.apply_tfms(tfms[0], size=512)
        B=mask.apply_tfms(tfms[0], do_resolve=False, size=512)
        A=A.data.numpy()
        A=np.transpose(A,(1,2,0))
        A = A.astype(np.float16)
        B=B.data.numpy()
        B=np.transpose(B,(1,2,0))
        B = B.astype(np.uint8)
        plt.imsave(os.path.join(img_png, '{}_{:03d}.png'.format(pid,i)),A)
        cv2.imwrite(os.path.join(gt_png, '{}_{:03d}_P.png'.format(pid,i)),B[:,:,0])

def pp_to_png(inputs):
  ix,path=inputs
  pid = path.split('/')[-1]
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

  roi_arr = cv_arr(gt_path + '/' + pid)
  roi_arr = roi_arr.astype(np.uint8)

  ex=[i for i in range(roi_arr.shape[2]) if np.max(roi_arr[:,:,i])==0] 
  cnt=len(ex) //4 
  ii=random.sample(ex, cnt)

  # save data 
  for i in range (roi_arr.shape[2]):
    if np.max(roi_arr[:,:,i])>0:
      limg=np.zeros((img_arr.shape[1],img_arr.shape[0],3),dtype=np.float16) 
      cimg=np.zeros((img_arr.shape[1],img_arr.shape[0]),dtype=np.uint8)
      limg[:,:,0] = flip_rot(img_arr[:,:,i]) 
      limg[:,:,1] = flip_rot(img_arr1[:,:,i])
      limg[:,:,2] = flip_rot(img_arr2[:,:,i]) 
      cimg = np.fliplr(np.rot90(roi_arr[:,:,i]))
      ip=os.path.join(img_png, '{}_{:03d}.png'.format(pid[:-4], i))
      gp=os.path.join(gt_png, '{}_{:03d}_P.png'.format(pid[:-4], i))
      plt.imsave(ip,limg)
      cv2.imwrite(gp,cimg)
  
  for i in ii:
       limg=np.zeros((img_arr.shape[1],img_arr.shape[0],3),dtype=np.float16)
       cimg=np.zeros((img_arr.shape[1],img_arr.shape[0]),dtype=np.uint8)
       limg[:,:,0] = flip_rot(img_arr[:,:,i]) 
       limg[:,:,1] = flip_rot(img_arr1[:,:,i])
       limg[:,:,2] = flip_rot(img_arr2[:,:,i])
       cimg = np.fliplr(np.rot90(roi_arr[:,:,i]))
       ip=os.path.join(img_png, '{}_{:03d}.png'.format(pid[:-4], i))
       gp=os.path.join(gt_png, '{}_{:03d}_P.png'.format(pid[:-4], i))
       plt.imsave(ip,limg)
       cv2.imwrite(gp,cimg)

"""
Make preprocessed train data
"""
img_png=os.path.join('/data/volume', 'img') 
gt_png=os.path.join('/data/volume', 'lbl')
if not os.path.exists(img_png): 
  os.mkdir(img_png)
if not os.path.exists(gt_png):
  os.mkdir(gt_png)

img_paths = [os.path.join(img_path, ii) for ii in training_img]
gt_paths= [os.path.join(gt_path, ii) for ii in training_gt]

lb0 = [];lb1=[];lb2=[];lb3=[]

for a in gt_paths:
    gtpath = a
    pid = gtpath.split('/')[-1] 

    roi_arr = cv_arr(gtpath)
    for i in range (roi_arr.shape[2]):
      lst=np.unique(roi_arr[:,:,i])
      name='{}_{:03d}'.format(pid[:-4], i)
      if 1 in lst:
        lb1.append(name)
      elif 2 in lst:
        lb2.append(name)
      elif 3 in lst:
        lb3.append(name)
      else:
        lb0.append(name)
# save preprocessed data
for input in enumerate(img_paths):
   pp_to_png(input)

a=len(lb0)//4
b=len(lb1)
c=len(lb2)
d=len(lb3)

# add tranformed data
augmentation(lb1,((a-b)//b)+1) 
augmentation(lb2,((a-c)//c)+1)
augmentation(lb3,((a-d)//d)+1) 