from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
import torch
import nibabel as nib
import os
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt

'''
GPU usage check
'''
print('gpu check')
print(torch.cuda.is_available())
print('gpu device')
print(torch.cuda.get_device_name(0))

'''
Load preprocessed dataset
--------------------------------
'''
path = Path('/data/volume')

path_lbl = path/'lbl'
path_img = path/'img'

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
if free > 8200: bs=16
else:           bs=8

src = (SegmentationItemList.from_folder(path_img)
       .split_by_rand_pct(valid_pct=0.2)
       .label_from_func(get_y_fn, classes=codes))

data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))


'''
Training
--------------------------------
total epoch: 10
--------------------------------
1st: lr = 1e-4    
** after unfreeze
2st: lr = slice(1e-3/400, 1e-3/4)
'''
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
learn.loss_func = CrossEntropyFlat(axis=1, weight = Tensor([1.,2.,3.,5.]).cuda())

lr=1e-4
learn.fit_one_cycle(5, slice(lr), pct_start=0.5)
learn.unfreeze()
lrs = slice(lr/400,lr/4)
learn.fit_one_cycle(5, lrs, pct_start=0.5)

# model save
model_pth = '/data/volume/model'
if not os.path.exists(model_pth):
    os.mkdir(model_pth)
learn.save('/data/volume/model/trained')

