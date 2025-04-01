# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 14:47:35 2021

@author: simonsanca
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print(matplotlib.rcParams['font.family'])
'''
# set file path
coco = pd.read_csv('D:/Clanek_GV/Grafi_python/rgb_coco_heads_100.csv', delimiter=';')
coco.head()

imgnet = pd.read_csv('D:/Clanek_GV/Grafi_python/rgb_imagenet_heads_100.csv', delimiter=';')
imgnet.head()
'''

#%%

# TRAIN LOSS - COCO in IMAGENET
# HEADS

# Geodetski vestnik
# visina lista = 20 cm
# sirina list = 14 cm

# set file path
fpath1 = 'E:/Clanek_GV/Grafi_python/rgb_coco_heads.csv'
coco = pd.read_csv(fpath1, delimiter=';')
coco.head()

fpath2 = 'E:/Clanek_GV/Grafi_python/rgb_imagenet_heads.csv'
imgnet = pd.read_csv(fpath2, delimiter=';')
imgnet.head()
fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,sharex=True,sharey=False,constrained_layout=True)
fig1.set_figheight(7.8)
fig1.set_figwidth(5.39)
#fig1.suptitle('Funkcije izgube učenja M1 in M2, 100 epoh, glavni sloji', fontsize=11)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 101, 10)
minor_ticks = np.arange(0, 101, 5)
for ax in ax1, ax2, ax3, ax4, ax5:
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.9)
    
# 1) LOSS
ax1.set_title('skupna funkcija izgube', fontsize=10)
ax1.set_ylim(0.0, 3) # sprememba skale pri grafih
#ax1.set_ylabel('loss')
# plot COCO
#ax1.scatter(coco['epoch'], coco['loss'], marker='.', color='g')
ax1.plot(coco['epoch'], coco['loss'], color='g', label='M1: COCO')
# plot IMAGENET
#ax1.scatter(imgnet['epoch'],imgnet['loss'],marker='.',color='b')
ax1.plot(imgnet['epoch'],imgnet['loss'],color='b',label='M2: IMAGENET')
ax1.grid('both')
# ----------------------------------------------------------------------------
# 2) RPN_CLASS_LOSS
ax2.set_title('RPN razred (angl. region proposal network)', fontsize=10)
ax2.set_ylim(0, 0.06)
#ax2.set_ylabel('rpn_loss')
# plot COCO
#ax2.scatter(coco['epoch'],coco['rpn_class_loss'],marker='.',color='g')
ax2.plot(coco['epoch'],coco['rpn_class_loss'],color='g',label='MS COCO')
# plot IMAGENET
#ax2.scatter(imgnet['epoch'],imgnet['rpn_class_loss'],marker='.',color='b')
ax2.plot(imgnet['epoch'],imgnet['rpn_class_loss'],color='b',label='IMAGENET')
ax2.grid('both')
# ----------------------------------------------------------------------------
# 3) MRCNN_CLASS_LOSS

# SAVITZKY-GOLAY FILTER
import scipy.signal as ss
ax3.set_title('Mask R-CNN razred', fontsize=10)
ax3.set_ylim(0, 0.3)
#ax3.set_ylabel('class_loss')
# plot COCO
wl = 7
po = 2
#ax3.scatter(coco['epoch'],coco['mrcnn_class_loss'],marker='.',color='g')
ax3.plot(coco['epoch'],ss.savgol_filter(coco['mrcnn_class_loss'],wl,po),color='g',label='MS COCO')
# plot IMAGENET
#ax3.scatter(imgnet['epoch'],imgnet['mrcnn_class_loss'],marker='.',color='b')
ax3.plot(imgnet['epoch'],ss.savgol_filter(imgnet['mrcnn_class_loss'],wl,po),color='b',label='IMAGENET')
ax3.grid('both')
# ----------------------------------------------------------------------------
# 4) MRCNN_BBOX LOSS
ax4.set_title('mejno okno (angl. bounding box)', fontsize=10)
ax4.set_ylim(0,1)
#ax4.set_ylabel('bbox_loss')
# plot COCO
#ax4.scatter(coco['epoch'],coco['mrcnn_bbox_loss'],marker='.',color='g')
ax4.plot(coco['epoch'],coco['mrcnn_bbox_loss'],color='g',label='MS COCO')
# plot IMAGENET
#ax4.scatter(imgnet['epoch'],imgnet['mrcnn_bbox_loss'],marker='.',color='b')
ax4.plot(imgnet['epoch'],imgnet['mrcnn_bbox_loss'],color='b',label='IMAGENET')
ax4.grid('both')
# ----------------------------------------------------------------------------
# 5) MRCNN_MASK_LOSS
ax5.set_title('maska', fontsize=10)
ax5.set_xlabel('# epoh', fontsize=10)
ax5.set_ylim(0,1)
#ax5.set_ylabel('mask_loss')
# plot COCO
#ax5.scatter(coco['epoch'],coco['mrcnn_mask_loss'],marker='.',color='g')
ax5.plot(coco['epoch'],coco['mrcnn_mask_loss'],color='g', label='MS COCO')
# plot IMAGENET
#ax5.scatter(imgnet['epoch'],imgnet['mrcnn_mask_loss'],marker='.',color='b')
ax5.plot(imgnet['epoch'],imgnet['mrcnn_mask_loss'],color='b',label='ImageNet')
ax5.grid('both')
ax5.legend(loc='lower center',
           bbox_to_anchor=(0.5, -0.75),
           frameon=False,
           fancybox=False,
           shadow=False,
           borderaxespad=0.,
           fontsize='medium',
           ncol=2)

# adjust bottom for legend
#fig1.subplots_adjust(bottom=0.25)
# ----------------------------------------------------------------------------

# save plot
fig1.savefig('graf_1_rgb.jpg', dpi=300, format='jpg')

#%%

# TRAIN LOSS - COCO in IMAGENET
# ALL - finetuned

# set file path
fpath1 = 'E:/Clanek_GV/Grafi_python/rgb_coco_all.csv'
coco = pd.read_csv(fpath1, delimiter=';')
coco.head()

fpath2 = 'E:/Clanek_GV/Grafi_python/rgb_imagenet_all.csv'
imgnet = pd.read_csv(fpath2, delimiter=';')
imgnet.head()

fig2, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,sharex=True,sharey=False,constrained_layout=True)
fig2.set_figheight(7.87)
fig2.set_figwidth(5.39)
#fig2.suptitle('Funkcije izgube učenja M3 in M4, 100 epoh, vsi sloji')

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 101, 10)
minor_ticks = np.arange(0, 101, 5)
for ax in ax1, ax2, ax3, ax4, ax5:
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.9)

# 1) LOSS
ax1.set_title('skupna funkcija izgube', fontsize=10)
ax1.set_ylim(0.0, 3) # sprememba skale pri grafih
#ax1.set_ylabel('loss')
# plot COCO
#ax1.scatter(coco['epoch'], coco['loss'], marker='.', color='g')
ax1.plot(coco['epoch'], coco['loss'], color='g', label='MS COCO')
# plot IMAGENET
#ax1.scatter(imgnet['epoch'],imgnet['loss'],marker='.',color='b')
ax1.plot(imgnet['epoch'],imgnet['loss'],color='b',label='IMAGENET')
ax1.grid('both')

# ----------------------------------------------------------------------------
# 2) RPN_CLASS_LOSS
ax2.set_title('RPN razred (angl. region proposal network)', fontsize=10)
ax2.set_ylim(0, 0.06)
#ax2.set_ylabel('rpn_loss')
# plot COCO
#ax2.scatter(coco['epoch'],coco['rpn_class_loss'],marker='.',color='g')
ax2.plot(coco['epoch'],coco['rpn_class_loss'],color='g',label='MS COCO')
# plot IMAGENET
#ax2.scatter(imgnet['epoch'],imgnet['rpn_class_loss'],marker='.',color='b')
ax2.plot(imgnet['epoch'],imgnet['rpn_class_loss'],color='b',label='ImageNet')
ax2.grid('both')
# ----------------------------------------------------------------------------
# 3) MRCNN_CLASS_LOSS
import scipy.signal as ss
ax3.set_title('Mask R-CNN razred', fontsize=10)
ax3.set_ylim(0, 0.3)
#ax3.set_ylabel('class_loss')
# plot COCO
wl = 3
po = 2
#ax3.scatter(coco['epoch'],coco['mrcnn_class_loss'],marker='.',color='g')
ax3.plot(coco['epoch'],ss.savgol_filter(coco['mrcnn_class_loss'],wl,po),color='g',label='MS COCO')
# plot IMAGENET
#ax3.scatter(imgnet['epoch'],imgnet['mrcnn_class_loss'],marker='.',color='b')
ax3.plot(imgnet['epoch'],ss.savgol_filter(imgnet['mrcnn_class_loss'],wl,po),color='b',label='IMAGENET')
ax3.grid('both')
# ----------------------------------------------------------------------------
# 4) MRCNN_BBOX_LOSS
ax4.set_title('mejno okno (angl. bounding box)', fontsize=10)
ax4.set_ylim(0,1)
#ax4.set_ylabel('bbox_loss')
# plot COCO
#ax4.scatter(coco['epoch'],coco['mrcnn_bbox_loss'],marker='.',color='g')
ax4.plot(coco['epoch'],coco['mrcnn_bbox_loss'],color='g',label='MS COCO')
# plot IMAGENET
#ax4.scatter(imgnet['epoch'],imgnet['mrcnn_bbox_loss'],marker='.',color='b')
ax4.plot(imgnet['epoch'],imgnet['mrcnn_bbox_loss'],color='b',label='IMAGENET')
ax4.grid('both')
# ----------------------------------------------------------------------------
# 5) MRCNN_MASK_LOSS
ax5.set_title('maska', fontsize=10)
ax5.set_xlabel('# epoh', fontsize=10)
ax5.set_ylim(0,1)
#ax5.set_ylabel('mask_loss')
# plot COCO
#ax5.scatter(coco['epoch'],coco['mrcnn_mask_loss'],marker='.',color='g')
ax5.plot(coco['epoch'],coco['mrcnn_mask_loss'],color='g', label='MS COCO')
# plot IMAGENET
#ax5.scatter(imgnet['epoch'],imgnet['mrcnn_mask_loss'],marker='.',color='b')
ax5.plot(imgnet['epoch'],imgnet['mrcnn_mask_loss'],color='b',label='ImageNet')
ax5.grid('both')
# ----------------------------------------------------------------------------
ax5.legend(loc='lower center',
           bbox_to_anchor=(0.5, -0.75),
           frameon=False,
           fancybox=False,
           shadow=False,
           borderaxespad=0.,
           fontsize='medium',
           ncol=2)
# ----------------------------------------------------------------------------
# save plot
fig2.savefig('graf_2_rgb.jpg', dpi=300, format='jpg')


#%%

## TO NE RABIMO 

# VALIDATION LOSS - COCO in IMAGENET
# HEADS
# zglajen vall loss

# set file path
fpath1 = 'C:/Users/simon/Documents/Clanek_GV/Grafi_python/rgb_coco_heads.csv'
coco = pd.read_csv(fpath1, delimiter=';')
coco.head()

fpath2 = 'C:/Users/simon/Documents/Clanek_GV/Grafi_python/rgb_imagenet_heads.csv'
imgnet = pd.read_csv(fpath2, delimiter=';')
imgnet.head()

# SAVITZKY-GOLAY FILTER
import scipy.signal as ss

# graf

fig3, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,sharex=True,sharey=False,constrained_layout=True)
fig3.set_figheight(7.87)
fig3.set_figwidth(5.39)
fig3.suptitle('Funkcije izgube validacije RGB M1 in M2, 100 epoh, glavni sloji')

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 101, 10)
minor_ticks = np.arange(0, 101, 5)
for ax in ax1, ax2, ax3, ax4, ax5:
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.9)
# ----------------------------------------------------------------------------
# 1) LOSS
ax1.set_title('skupna funkcija izgube', fontsize=10)
#ax1.set_ylabel('loss')
# plot COCO
#ax1.scatter(coco['epoch'], coco['loss'], marker='.', color='g')
wl = 7
po = 2
ax1.plot(coco['epoch'], ss.savgol_filter(coco['val_loss'],wl,po), color='g', label='MS COCO')
# plot IMAGENET
#ax1.scatter(imgnet['epoch'],imgnet['loss'],marker='.',color='b')
ax1.plot(imgnet['epoch'],ss.savgol_filter(imgnet['val_loss'],wl,po),color='b',label='IMAGENET')
# ----------------------------------------------------------------------------
# 2) RPN_CLASS_LOSS
ax2.set_title('RPN razred (angl. region proposal network)', fontsize=10)
#ax2.set_ylabel('rpn_loss')
# plot COCO
wl = 7
po = 2
ax2.plot(coco['epoch'], ss.savgol_filter(coco['val_rpn_class_loss'],wl,po), color='g', label='MS COCO')
# plot IMAGENET
#ax1.scatter(imgnet['epoch'],imgnet['loss'],marker='.',color='b')
ax2.plot(imgnet['epoch'],ss.savgol_filter(imgnet['val_rpn_class_loss'],wl,po),color='b',label='IMAGENET')
# ----------------------------------------------------------------------------
# 3) MRCNN_CLASS_LOSS
ax3.set_title('Mask R-CNN razred', fontsize=10)
#ax3.set_ylabel('class_loss')
# plot COCO
wl = 7
po = 2
#ax3.scatter(coco['epoch'],coco['mrcnn_class_loss'],marker='.',color='g')
ax3.plot(coco['epoch'],ss.savgol_filter(coco['val_mrcnn_class_loss'],wl,po),color='g',label='MS COCO')
# plot IMAGENET
#ax3.scatter(imgnet['epoch'],imgnet['mrcnn_class_loss'],marker='.',color='b')
ax3.plot(imgnet['epoch'],ss.savgol_filter(imgnet['val_mrcnn_class_loss'],wl,po),color='b',label='IMAGENET')
# ----------------------------------------------------------------------------
# 4) MRCNN_BBOX_LOSS
ax4.set_title('mejno okno (angl. bounding box)', fontsize=10)
#ax4.set_ylabel('bbox_loss')
# plot COCO
wl = 7
po = 2
#ax4.scatter(coco['epoch'],coco['mrcnn_bbox_loss'],marker='.',color='g')
ax4.plot(coco['epoch'],ss.savgol_filter(coco['val_mrcnn_bbox_loss'],wl,po),color='g',label='MS COCO')
# plot IMAGENET
#ax4.scatter(imgnet['epoch'],imgnet['mrcnn_bbox_loss'],marker='.',color='b')
ax4.plot(imgnet['epoch'],ss.savgol_filter(imgnet['val_mrcnn_bbox_loss'],wl,po),color='b',label='IMAGENET')
# ----------------------------------------------------------------------------
# 5) MRCNN_MASK_LOSS
ax5.set_title('maska', fontsize=10)
ax5.set_xlabel('# epoh', fontsize=10)
#ax5.set_ylabel('mask_loss')
# plot COCO
wl = 7
po = 2
#ax5.scatter(coco['epoch'],coco['mrcnn_mask_loss'],marker='.',color='g')
ax5.plot(coco['epoch'],ss.savgol_filter(coco['val_mrcnn_mask_loss'],wl,po), color='g', label='MS COCO')
# plot IMAGENET
#ax5.scatter(imgnet['epoch'],imgnet['mrcnn_mask_loss'],marker='.',color='b')
ax5.plot(imgnet['epoch'],ss.savgol_filter(imgnet['val_mrcnn_mask_loss'],wl,po), color='b',label='ImageNet')
# ----------------------------------------------------------------------------
ax5.legend(loc='lower center',
           bbox_to_anchor=(0.5, -0.75),
           frameon=False,
           fancybox=False,
           shadow=False,
           borderaxespad=0.,
           fontsize='medium',
           ncol=2)

fig3.savefig('graf_3_rgb.jpg', dpi=300, format='jpg')

#%%

# VALIDATION LOSS - COCO in IMAGENET
# ALL - finetune
# zglajen vall loss

# set file path
fpath1 = 'C:/Users/simon/Documents/Clanek_GV/Grafi_python/rgb_coco_all.csv'
coco = pd.read_csv(fpath1, delimiter=';')
coco.head()

fpath2 = 'C:/Users/simon/Documents/Clanek_GV/Grafi_python/rgb_imagenet_all.csv'
imgnet = pd.read_csv(fpath2, delimiter=';')
imgnet.head()

# SAVITZKY-GOLAY FILTER
import scipy.signal as ss

# graf
fig4, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,sharex=True,sharey=False,constrained_layout=True)
fig4.set_figheight(7.87)
fig4.set_figwidth(5.39)
fig4.suptitle('Funkcije izgube validacije RGB M3 in M4, 100 epoh, vsi sloji')

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 101, 10)
minor_ticks = np.arange(0, 101, 5)
for ax in ax1, ax2, ax3, ax4, ax5:
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.9)
# ----------------------------------------------------------------------------
# 1) LOSS
ax1.set_title('skupna funkcija izgube', fontsize=10)
#ax1.set_ylabel('loss')
# plot COCO
#ax1.scatter(coco['epoch'], coco['loss'], marker='.', color='g')
wl = 7
po = 2
ax1.plot(coco['epoch'], ss.savgol_filter(coco['val_loss'],wl,po), color='g', label='MS COCO')
# plot IMAGENET
#ax1.scatter(imgnet['epoch'],imgnet['loss'],marker='.',color='b')
ax1.plot(imgnet['epoch'],ss.savgol_filter(imgnet['val_loss'],wl,po),color='b',label='IMAGENET')
# ----------------------------------------------------------------------------
# 2) RPN_CLASS_LOSS
ax2.set_title('RPN razred (angl. region proposal network)', fontsize=10)
#ax2.set_ylabel('rpn_loss')
# plot COCO
wl = 7
po = 2
ax2.plot(coco['epoch'], ss.savgol_filter(coco['val_rpn_class_loss'],wl,po), color='g', label='MS COCO')
# plot IMAGENET
#ax1.scatter(imgnet['epoch'],imgnet['loss'],marker='.',color='b')
ax2.plot(imgnet['epoch'],ss.savgol_filter(imgnet['val_rpn_class_loss'],wl,po),color='b',label='IMAGENET')
# ----------------------------------------------------------------------------
# 3) MRCNN_CLASS_LOSS
ax3.set_title('Mask R-CNN razred', fontsize=10)
#ax3.set_ylabel('class_loss')
# plot COCO
wl = 7
po = 2
#ax3.scatter(coco['epoch'],coco['mrcnn_class_loss'],marker='.',color='g')
ax3.plot(coco['epoch'],ss.savgol_filter(coco['val_mrcnn_class_loss'],wl,po),color='g',label='MS COCO')
# plot IMAGENET
#ax3.scatter(imgnet['epoch'],imgnet['mrcnn_class_loss'],marker='.',color='b')
ax3.plot(imgnet['epoch'],ss.savgol_filter(imgnet['val_mrcnn_class_loss'],wl,po),color='b',label='IMAGENET')
# ----------------------------------------------------------------------------
# 4) MRCNN_BBOX_LOSS
ax4.set_title('mejno okno (angl. bounding box)', fontsize=10)
#ax4.set_ylabel('bbox_loss')
# plot COCO
wl = 7
po = 2
#ax4.scatter(coco['epoch'],coco['mrcnn_bbox_loss'],marker='.',color='g')
ax4.plot(coco['epoch'],ss.savgol_filter(coco['val_mrcnn_bbox_loss'],wl,po),color='g',label='MS COCO')
# plot IMAGENET
#ax4.scatter(imgnet['epoch'],imgnet['mrcnn_bbox_loss'],marker='.',color='b')
ax4.plot(imgnet['epoch'],ss.savgol_filter(imgnet['val_mrcnn_bbox_loss'],wl,po),color='b',label='IMAGENET')
# ----------------------------------------------------------------------------
# 5) MRCNN_MASK_LOSS
ax5.set_title('maska', fontsize=10)
ax5.set_xlabel('# epoh', fontsize=10)
#ax5.set_ylabel('mask_loss')
# plot COCO
wl = 7
po = 2
#ax5.scatter(coco['epoch'],coco['mrcnn_mask_loss'],marker='.',color='g')
ax5.plot(coco['epoch'],ss.savgol_filter(coco['val_mrcnn_mask_loss'],wl,po), color='g', label='MS COCO')
# plot IMAGENET
#ax5.scatter(imgnet['epoch'],imgnet['mrcnn_mask_loss'],marker='.',color='b')
ax5.plot(imgnet['epoch'],ss.savgol_filter(imgnet['val_mrcnn_mask_loss'],wl,po), color='b',label='ImageNet')
# ----------------------------------------------------------------------------
ax5.legend(loc='lower center',
           bbox_to_anchor=(0.5, -0.75),
           frameon=False,
           fancybox=False,
           shadow=False,
           borderaxespad=0.,
           fontsize='medium',
           ncol=2)

fig3.savefig('graf_4_rgb.jpg', dpi=300, format='jpg')

#%%
