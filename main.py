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
    c.rotateCoords('y', -3)
    # c.matrixTransform([
    #     [1.1,0,0],
    #     [0,1,0],
    #     [0,0,1]
    # ])
    #c.rotateCoords('z', 1)
    c.update2dCoords()
    #sleep(0.0)

def pretransformation():
    c.scaleCoords(-4)
    # c.rotateCoords('x',-90)
    # c.shiftCoords('y', 100)



@numba.njit() #I LOVE NUMBA; IT MADE THE CODE SO MUCH QUICKER AND GOT RID OF ALL THE SILLY NUMPY STUFF ITS SO SIMPLE NOW, I OWE TRAVIS OLIPHANT MY LIFE
def rasterize_gouraud(coords, view,zbuffer,av_normals,coords_3d,normal):
    A, B, C = coords
    n1,n2,n3 = av_normals
    x1, y1, z1 = coords_3d[0]
    x2, y2, z2 = coords_3d[1]
    x3, y3, z3 = coords_3d[2]
    a = (z1-z3) * (y2 - y3) - (y1-y3) * (z2 - z3)
    b = -((x2-x3)*(z1-z3)-(z2-z3)*(x1-x3))
    c = (x2-x3)*(y1-y3)-(y2-y3)*(x1-x3)
    d = -(a*x1+b*y1+c*z1)

    #Used for the bounding box
    min_x = max(int(min(A[0], B[0], C[0])), 0)
    max_x = min(int(max(A[0], B[0], C[0])) + 1, view.shape[1])
    min_y = max(int(min(A[1], B[1], C[1])), 0)
    max_y = min(int(max(A[1], B[1], C[1])) + 1, view.shape[0])

    #area of triangle
    area = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
    if area == 0 or abs(c)<=0:
        return
    colors = []
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):

            #barycentric coordinates are the goat
            w0 = (B[0] - A[0]) * (y - A[1]) - (B[1] - A[1]) * (x - A[0])
            w1 = (C[0] - B[0]) * (y - B[1]) - (C[1] - B[1]) * (x - B[0])
            w2 = (A[0] - C[0]) * (y - C[1]) - (A[1] - C[1]) * (x - C[0])

            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                Z = (a * x + b * y + d) / -c
                if ZBUFF:

                    if Z >= zbuffer[y, x]:
                        continue
                    zbuffer[y, x] = Z

                P = np.array([x, y])

                v1,v2 = A-B,C-B
                total_area = 0.5 * abs(v1[0]*v2[1]-v1[1]*v2[0])

                v1, v2 = B-P, C - P
                alpha = 0.5 * abs(v1[0]*v2[1]-v1[1]*v2[0])

                v1, v2 = C - P, A - P
                beta = 0.5 * abs(v1[0]*v2[1]-v1[1]*v2[0])

                v1, v2 = A - P, B - P
                gamma = 0.5 * abs(v1[0]*v2[1]-v1[1]*v2[0])
                alpha/=total_area
                beta/=total_area
                gamma/=total_area
                s = alpha+beta+gamma
                alpha/=s
                beta/=s
                gamma/=s

                #shitty manual dot product because numba isn't fond of numpy functions
                diffuse = 255*(max(n1.dot(LIGHT_VECTOR),0)*alpha+max(n2.dot(LIGHT_VECTOR),0)*beta+ max(n3.dot(LIGHT_VECTOR),0)*gamma)
                tolight = np.array([-x,-y,-Z])
                tolight=tolight/np.linalg.norm(tolight)
                R= 2*normal.dot(tolight)*normal-tolight
                R=R/np.linalg.norm(R)
                specular = max(0,REFLECTIVITY_CONSTANT*R.dot(tolight)**PHONG_EXPONENT)


                total_intensity = diffuse+AMBIENT_INTENSITY+specular
                total_intensity = 255*(total_intensity/255)**GAMMA

                color = np.array(3*[min(255,total_intensity)])
                colors.append(color)
                view[y, x] = color
    # print(colors)







def display(shape):
    global view
    view = np.zeros((HEIGHT, WIDTH, 3),dtype=np.uint8)
    zbuffer = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float64)
    centArray = np.array([WIDTH/2,HEIGHT/2])
    # print(len(shape.validFaces()))
    for face in shape.validFaces():
        coords = face.TwoDCoords+centArray
        # color = shader(face)
        # color = np.clip(color, 0, 255).astype(np.uint8)
        # rasterize(coords,color,view)
        rasterize_gouraud(coords,view,zbuffer,face.avNorms,face.points,face.normal)






if __name__ == '__main__':

    c = OBJFile("models/Shambler.obj",reverseNormals=False,loadAverageNorms=True)
    pretransformation()
    c.update2dCoords()
    shader = Lambertian()

    while cv.waitKey(20)&0xff != ord('x'):
        view = np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8)

        if BENCHMARK: now = time()

        display(c)
        cv.imshow("3d Render",view)

        if BENCHMARK: print(f"Displaying:",time()-now,"seconds"); now = time()

        TransformationLoop()
        c.update2dCoords()

        if BENCHMARK: print("Transforming:",time()-now,"seconds")