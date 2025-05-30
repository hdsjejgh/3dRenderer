import numpy as np
import math

WIDTH = 800
HEIGHT = 800
FPS = 30
FOV = 200
VIEW_VECTOR = np.array([0,0,1],dtype = 'float64')
LIGHT_VECTOR = np.array([0,0,1], dtype='float64')
REFLECTION_VECTOR = np.array([1/math.sqrt(3),1/math.sqrt(3),-1/math.sqrt(3)],dtype='float64')
PHONG_EXPONENT = 100
REFLECTIVITY_CONSTANT = 0.7
AMBIENT_INTENSITY = 0.1*255
ZBUFF = False
GAMMA = 0.8 #for gamma correction

