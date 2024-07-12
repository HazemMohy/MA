import numpy as np

def create_overlay(pred, target):
    pred    = np.squeeze(pred)
    target  = np.squeeze(target)
    target[target > 1] = 1 
    pred[pred > 1] = 1 

    res = target*2 + pred
    res[res==2] = 10
    res[res==3] = 2 
    res[res==10]= 3
    return res
