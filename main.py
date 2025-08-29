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


Model = OBJFile("models/Shambler.obj",reverseNormals=True,texture="textures/Shambler.png")


def TransformationLoop():
    Model.rotateCoords('x', 4)
    Model.rotateCoords('y', -3)
    # c.matrixTransform([
    #     [1.1,0,0],
    #     [0,1,0],
    #     [0,0,1]
    # ])
    Model.rotateCoords('z', 7)
    Model.update2dCoords()


def pretransformation():
    Model.scaleCoords(-3)
    Model.rotateCoords('x',-90)
    # c.shiftCoords('y', -75)


def display_phong(shape):

    global view
    view = np.zeros((HEIGHT, WIDTH, 3),dtype=np.uint8)
    zbuffer = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float64)
    centArray = np.array([WIDTH/2,HEIGHT/2])
    for face in shape.validFaces():

        coords = face.TwoDCoords+centArray

        rasterize_phong(coords,view,zbuffer,face.avNorms,face.points,face.normal,(255,255,255))



def display_phong_textured(shape):
    if shape.textured:
        t = shape.texture
    else:
        t = False
    global view
    view = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    zbuffer = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float64)
    centArray = np.array([WIDTH / 2, HEIGHT / 2])
    # print(len(shape.validFaces()))
    for face in shape.validFaces():
        if t is not False:
            tt = face.texturepoints
        else:
            tt = None
        coords = face.TwoDCoords + centArray
        # color = shader(face)
        # color = np.clip(color, 0, 255).astype(np.uint8)
        # rasterize(coords,color,view)
        rasterize_phong_texture(coords, view, zbuffer, face.avNorms, face.points, face.normal, (255, 255, 255), texturecoords=tt, texture=t)

if __name__ == '__main__':

    t = time()
    pretransformation()
    Model.update2dCoords()
    shader = Lambertian()

    print(f"Prefunctioning: {time() - t} seconds")
    t = time()

    while cv.waitKey(20)&0xff != ord('x'):
        view = np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8)




        display_phong_textured(Model)
        cv.imshow("3d Render",view)



        TransformationLoop()
        Model.update2dCoords()

