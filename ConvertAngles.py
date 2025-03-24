#!/usr/bin/python
# -*- coding: utf-8 -*-

import math

def dms2dms(dms,ndec):

    '''Funkcija dms2dms iz oblike kota podanega v obliki;
       dms (SSS.MMSSSSSSS) naredi niz znakov v obliki SS MM SS.SSSS'''
    if dms < 0:
        kot = -dms
    else:
        kot = dms
        
    stopinje = math.floor(kot)
    minute = math.floor((kot-stopinje)*100.0)
    sekunde = ((kot-stopinje)*100.0 - minute)*100.0
    
    if abs(sekunde-100) < 0.001:
        sekunde = 0
        minute = minute + 1
        
    if minute == 60:
        minute = 0
        stopinje = stopinje + 1
    
    if dms < 0:
        stopinje = -stopinje
        
    stopinje = int(round(stopinje)) 
    minute = int(round(minute)) 
    sekunde = int(round(sekunde))
    
    print(stopinje,'',minute,'',sekunde,'')
    
    return

def dms2deg(dms):
    if dms < 0.0:
        kot = abs(dms)
    else:
        kot = dms
    
    stopinje = math.floor(kot)
    minute = math.floor((kot-stopinje)*100.0)
    sekunde = ((kot-stopinje)*100.0 - minute)*100.0
    
    if abs(sekunde - 100.0) < 1e-5:
        sekunde = 0.0
        minute = minute + 1
        
    deg = stopinje + minute/60.0 + sekunde/3600.0
    
    if dms < 0:
        deg = -deg
        
    return deg

def deg2dms(deg):
    
    if (deg < 0):
        kot = -deg
    else:
        kot = deg
        
    stopinje = math.floor(kot)
    minute = math.floor((kot-stopinje)*60.0)
    sekunde = ((kot-stopinje)*60.0 - minute)*60.0
    
    if abs(sekunde - 60.0) < 1e-10:
        sekunde = 0
        minute = minute + 1
        
    if minute == 60:
        minute = 0
        stopinje = stopinje + 1
    
    if abs(sekunde - 100) < 0.001:
        sekunde  = 0
        minute = minute + 1
    
    if minute == 60:
        minute = 0
        stopinje = stopinje + 1
    
    dms = stopinje + minute/100.0 + sekunde/10000.0
    
    return dms

def deg2gon(deg):
    gon = deg*(10.0/9.0)
    return gon

def deg2rad(deg):
    rad = deg*(math.pi/180.0)
    return rad

def dms2rad(dms):
    deg = dms2deg(dms)
    rad = deg2rad(deg)
    return rad

def gon2deg(gon):
    deg = gon*(9.0/10.0)
    return deg

def gon2rad(gon):
    rad = gon*(math.pi/200.0)
    return rad

def rad2deg(rad):
    deg = rad*(180.0/math.pi)
    return deg

def rad2dms(rad):
    deg = rad*(180.0/math.pi)
    dms = deg2dms(deg)
    return dms

def rad2gon(rad):
    gon = rad*(200.00/math.pi)
    return gon

def rad2sec(rad):
    sec = rad*(180.0/math.pi)*3600
    return sec

def sec2rad(sec):
    rad = (sec/3600)*math.pi/180
    return rad

#END
