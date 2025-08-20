import numpy as np
import math

WIDTH = 800
HEIGHT = 800
FPS = 30
FOV = 250
VIEW_VECTOR = np.array([0,0,1],dtype = 'float64')
CAMERA_POS = np.array([0,0,-50],dtype = 'float64')
LIGHT_POS = np.array([30,30,0],dtype = 'float64')
LIGHT_VECTOR = -LIGHT_POS/np.linalg.norm(LIGHT_POS)
# REFLECTION_VECTOR = np.array([1/math.sqrt(3),1/math.sqrt(3),-1/math.sqrt(3)],dtype='float64')
PHONG_EXPONENT = 10
REFLECTIVITY_CONSTANT = 1
AMBIENT_INTENSITY = 0.25*255
ZBUFF = False
GAMMA = 0.8 #for gamma correction


