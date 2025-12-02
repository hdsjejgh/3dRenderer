import parameters
from shapes import rot_matrix
import numpy as np



#Rotations the light source about a given access (x,y,z) by a given number of degrees
def lightRot(axis,deg,index,L_array):
    L_array[index][:] = rot_matrix(axis, deg) @ L_array[index]

#Scales the light position vector by given magnitude
def lightScale(magnitude,index,L_array):
    L_array[index]*=magnitude

def createLight(x,y,z,L_array):
    l = np.array([x,y,z],dtype=np.float32)
    L_array.append(l)