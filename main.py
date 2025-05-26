import math
from time import sleep,time
import numpy as np

from shapes import *
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from parameters import *
from shaders import *
import cv2 as cv
import numba

BENCHMARK = False

# c=Cube(
#         coords=[[-2,2,2],[2,2,2],[2,-2,2],[-2,-2,2],[-2,2,6],[2,2,6],[2,-2,6],[-2,-2,6],],
#          faces=[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2],[0,1,5,4],[2,3,7,6],]
#     )


def TransformationLoop():
    #c.rotateCoords('x', 4)
    #c.rotateCoords('y', -3)
    # c.matrixTransform([
    #     [1.1,0,0],
    #     [0,1,0],
    #     [0,0,1]
    # ])
    #c.rotateCoords('z', 1)
    c.update2dCoords()
    #sleep(0.0)

def pretransformation():
    c.scaleCoords(-2)
    # c.rotateCoords('x',-90)
    # c.shiftCoords('y', 100)



def display(shape,shader):
    global view
    view = np.zeros((HEIGHT, WIDTH, 3),dtype=np.uint8)
    centArray = np.array([WIDTH/2,HEIGHT/2])
    # print(len(shape.validFaces()))
    for face in shape.validFaces():
        coords = face.TwoDCoords+centArray
        # color = shader(face)
        # color = np.clip(color, 0, 255).astype(np.uint8)
        # rasterize(coords,color,view)
        rasterize_gouraud(coords,view,face.avNorms,face.points)






if __name__ == '__main__':

    c = OBJFile("models/Shambler.obj",reverseNormals=False,loadAverageNorms=True)
    pretransformation()
    c.update2dCoords()
    shader = Lambertian()

    while cv.waitKey(20)&0xff != ord('x'):
        view = np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8)

        if BENCHMARK: now = time()

        display(c,shader)
        cv.imshow("3d Render",view)

        if BENCHMARK: print(f"Displaying:",time()-now,"seconds"); now = time()

        TransformationLoop()
        c.update2dCoords()

        if BENCHMARK: print("Transforming:",time()-now,"seconds")