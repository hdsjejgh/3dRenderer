import parameters
from shapes import rot_matrix
import numpy as np

#Rotations the light source about a given access (x,y,z) by a given number of degrees
def lightRot(axis,deg):
    parameters.LIGHT_POS[:] = rot_matrix(axis, deg) @ parameters.LIGHT_POS
    parameters.LIGHT_VECTOR[:] = -parameters.LIGHT_POS / np.linalg.norm(parameters.LIGHT_POS)

#Scales the light position vector by given magnitude
def lightScale(magnitude):
    parameters.LIGHT_POS*=magnitude
    if magnitude<0:
        parameters.LIGHT_VECTOR *= -1