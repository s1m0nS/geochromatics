# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:20:47 2021

@author: simon
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

#-----------------------------------------------------------------------------------#

# set file path
fpath1 = 'C:/Users/simon/Documents/Clanek_GV/Grafi_python/coco_heads_100.csv'
data = pd.read_csv(fpath1, delimiter=';')
data.head()

fpath2 = 'C:/Users/simon/Documents/Clanek_GV/Grafi_python/imagenet_heads_100.csv'
imgnet = pd.read_csv(fpath2, delimiter=';')
imgnet.head()

# METRIKE SI SAM IZBERI ZA PLOT
# https://stackoverflow.com/questions/55360262/what-exactly-are-the-losses-in-matterport-mask-r-cnn
# print(data.head()) # -> da vidis opcije

# Izbrana metrika
#data['class_loss'] = data['rpn_class_loss'] + data['mrcnn_class_loss']
#data['val_class_loss'] = data['val_rpn_class_loss'] + data['val_mrcnn_class_loss']


#-----------------------------------------------------------------------------------#

# Get the data
x = np.asarray(data['epoch'])
y = np.asarray(data['loss'])
yv = np.asarray(data['val_loss'])

# 1. MOVING AVERAGE FILTER

filter_length = 25 # change to odd number

# training
tm_avg = np.convolve(y, np.ones(filter_length), mode='same')
tm_avg = tm_avg / filter_length # divide by filter_length

# validation
tmv = np.convolve(yv, np.ones(filter_length), mode='same')
tmv = tmv/filter_length

# plot moving average filter results by choice
#plt.figure(1)
#plt.plot(x,y,label='training')
#plt.plot(x,yv,label='validation')
#plt.plot(x,tm_avg,label='moving_average')
#plot.plot(x,tmv,label='moving_avg_val')
#plt.grid(True)
#plt.legend()

# 2. MEDIAN FILTER
from scipy.signal import medfilt

med_fil = medfilt(y, filter_length) # train
med_filv = medfilt(yv, filter_length) # val

# 3. SAVITZKY-GOLAY FILTER
import scipy.signal as ss

window_length = filter_length
sg_filtered_train = ss.savgol_filter(y,window_length,2)
sg_filtered_val = ss.savgol_filter(yv,window_length,2)

'''
plt.plot(x,y)
plt.plot(x,yv)
plt.plot(x,sg_filtered_train)
plt.plot(x,sg_filtered_val)
'''

# PLOT FILTERED DATA
# training

plt.figure(2)
plt.title('Line smoothing - training')
plt.plot(x, y, color='green', label='raw')
#plt.plot(x, tm_avg, color = 'black', label='moving_average')
#plt.plot(x, med_fil, color= 'red', label='median')
plt.plot(x, sg_filtered_train, color='magenta', label='Savitzky-Golay')
#plt.ylim([0,3])
plt.grid(True)
plt.legend()
#%%
# validation

plt.figure(3)
plt.title('Line smooting - validation')
plt.plot(x, yv, color='green', label='validation')
plt.plot(x, tmv, color = 'black', label='moving_average')
plt.plot(x, med_filv, color= 'red', label='median')
plt.plot(x, sg_filtered_val, color='magenta', label='Savitzky-Golay')
plt.grid(True)
plt.legend()


#%%

# 4. GAUSSIAN PROCESSES
from sklearn import gaussian_process
from sklearn import preprocessing

# https://scikit-learn.org/stable/modules/gaussian_process.htm
# https://stats.stackexchange.com/questions/169995/why-does-my-train-data-not-fall-in-confidence-interval-with-scikit-learn-gaussia/171220

X = np.array(x).reshape((len(x), 1))
y = np.array(y).reshape(-1,1)

#print(X)
#print(y)

scaler = preprocessing.MinMaxScaler().fit(y)
scaler.transform(y)

X, y = X, y
gp = gaussian_process.GaussianProcessRegressor()
gp.fit(X,y)
tV = np.array(yv).reshape(-1,1)
py, sigma = gp.predict(tV,return_std=True, return_cov=False)
conf_int = sigma * 1.9 # 95 confidence interval

# calculate confidence interval
ys = py.ravel() - conf_int
yz = py.ravel() + conf_int

# plot the result
plt.figure(4)
plt.title('Gaussian process regression - train/val class loss')
# fill area between points
plt.fill_between(x,
                 ys,
                 yz,
                 alpha = 0.85, # transparency
                 label = '95% confidence interval',
                 color='#B22222')
#plt.plot(x, ys, label='Prediction')
plt.scatter(x, y, label='Train')
plt.scatter(x, yv, label='Validaton')
plt.legend()
plt.grid()

#%%

# Gaussian process, confidence intervals
# https://app.dominodatalab.com/u/fonnesbeck/gp_showdown/view/GP+Showdown.ipynb

# SCIKIT LEARN
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import seaborn as sns

x = np.asarray(data['epoch'])
y = np.asarray(data['class_loss']) # training
yv = np.asarray(data['val_class_loss']) # valdiation

kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)

X = x.reshape(-1, 1)
print(X.shape)

# get kernel parameters
print(gp.kernel_)

gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
gp.fit(X, y)

x_pred = x.reshape(-1,1)
y_pred, sigma = gp.predict(x_pred, return_std=True)

sigma_train = sigma

plt.figure(5)
plt.title('Gaussian process fit - training')
plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
         np.concatenate([y_pred - 2*sigma,
                        (y_pred + 2*sigma)[::-1]]),
         alpha=.5, fc='grey', ec='None', label='95% Confidence interval')
sns.regplot(x, y, fit_reg=False, label='Data')
plt.plot(x_pred, y_pred, color='black', label='Prediction')
plt.xlabel('# epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend(loc='upper right')

#%% Validation

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import seaborn as sns


x = np.asarray(data['epoch'])
y = np.asarray(data['loss'])

yv = np.asarray(data['val_loss'])

kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)

X = x.reshape(-1, 1)
print(X.shape)

# get kernel parameterse
print(gp.kernel_)

gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
gp.fit(X, yv)

x_pred = x.reshape(-1,1)
y_pred, sigma = gp.predict(x_pred, return_std=True)

sigma_val = sigma

plt.figure(6)
plt.title('Gaussian process fit - validation')
plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
         np.concatenate([y_pred - 2*sigma,
                        (y_pred + 2*sigma)[::-1]]),
         alpha=.5, fc='grey', ec='None', label='95% Confidence interval')
sns.regplot(x, yv, fit_reg=False, label='Data')
plt.plot(x_pred, y_pred, color='black', label='Prediction')
plt.xlabel('# epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend(loc='upper right')

#%%

# DETERMINE THE EPOCH WITH MIN_DIFFERENCE BETWEEN TRAIN AND VAL

# calculate eucledian distance between two set of points
# https://file.biolab.si/textbooks/uozp/zapiski.pdf
# mere razlicnosti -> evklidska razdalja za ugotavljanje razlik med dvema skupinama podatkov

# data
x = np.asarray(data['epoch'])
y = np.asarray(data['loss'])

yv = np.asarray(data['val_loss'])

from scipy.spatial import distance

# tp - training points
# vp - validation points
#tp = np.column_stack((x,y))
#vp = np.column_stack((x,yv))

# scipy eucledian
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html#scipy.spatial.distance.euclidean

# the seljak way
dist = []
for e,t,v in zip(x,y,yv):
    dst = distance.euclidean(t, v)
    #print(e, dst)
    dist.append(round(dst,3))
    
#print(dist)

# create a dictionary with epoch and distance
x = np.ndarray.tolist(x)
edst = dict(zip(x,dist))

# calculate minimal distance at epoch between test and validation set
print('Minimal distance at epoch: ', min(edst.items(), key=lambda x: x[1]))

# visualize distances
y_pos = np.arange(len(x))
plt.figure(7)
plt.title('Eucledian distances between train and val')
plt.bar(y_pos, dist, align='center', alpha=0.8)
plt.xticks(y_pos, dist)
plt.ylabel('Eucledian distance train - val')
plt.xlabel('# epoch')
plt.xticks([])
plt.show()
    
#%%
