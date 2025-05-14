import numpy as np

WIDTH = 800
HEIGHT = 800
FPS = 30
FOV = 200
VIEW_VECTOR = np.array([0,0,1],dtype = 'float64')
VIEW_VECTOR /= np.linalg.norm(VIEW_VECTOR)
LIGHT_VECTOR = np.array([0,0,1], dtype='float64')
LIGHT_VECTOR /= np.linalg.norm(LIGHT_VECTOR)
DISPLAY_MODE = {0:'pygame', 1:'rasterizer'}[1]
BACKFACECULLING = True