import parameters
from shapes import rot_matrix
import numpy as np



#Rotations the light source about a given access (x,y,z) by a given number of degrees
def lightRot(axis,deg,index,LInfo_array):
    LInfo_array[0][index][:] = rot_matrix(axis, deg) @ LInfo_array[0][index]

#Scales the light position vector by given magnitude
def lightScale(magnitude,index,LInfo_array):
    LInfo_array[0][index]*=magnitude

def createLight(x,y,z,LInfo_array,intensity=1.0):
    l = np.array([x,y,z],dtype=np.float32)
    LInfo_array[0].append(l)
    LInfo_array[1].append(intensity)