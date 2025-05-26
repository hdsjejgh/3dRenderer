import numpy as np

WIDTH = 800
HEIGHT = 800
FPS = 30
FOV = 200
VIEW_VECTOR = np.array([0,0,1],dtype = 'float64')
VIEW_VECTOR /= np.linalg.norm(VIEW_VECTOR)
LIGHT_VECTOR = np.array([0,0,1], dtype='float64')
AMBIENT_INTENSITY = 0.3*255
