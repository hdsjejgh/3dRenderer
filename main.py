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
    c.rotateCoords('x', 4)
    c.rotateCoords('y', -3)
    # c.matrixTransform([
    #     [1.1,0,0],
    #     [0,1,0],
    #     [0,0,1]
    # ])
    c.rotateCoords('z', 7)
    c.update2dCoords()
    #sleep(0.0)

def pretransformation():
    c.scaleCoords(-100)
    c.rotateCoords('x',-90)
    # c.shiftCoords('y', -75)



@numba.njit()
def rasterize_phong_texture(coords, view, zbuffer, av_normals, coords_3d, normal,color = (255,255,255), texturecoords=None, texture=np.empty((1,1),np.int64)):
    base = np.array(color, dtype=np.float64)
    red,green,blue = base
    A, B, C = coords.astype(np.float64)
    n1, n2, n3 = [np.asarray(n, dtype=np.float64) for n in av_normals]
    coords_3d = coords_3d.astype(np.float64)
    base = np.array(color,dtype=np.float64)
    #The 3d coordinated
    x1, y1, z1 = coords_3d[0]
    x2, y2, z2 = coords_3d[1]
    x3, y3, z3 = coords_3d[2]

    #Used for the bounding box
    min_x = max(int(min(A[0], B[0], C[0])), 0)
    max_x = min(int(max(A[0], B[0], C[0])) + 1, view.shape[1])
    min_y = max(int(min(A[1], B[1], C[1])), 0)
    max_y = min(int(max(A[1], B[1], C[1])) + 1, view.shape[0])

    #area of the triangle face
    v1, v2 = A - B, C - B
    total_area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

    if total_area == 0: #if triangle is too small, just skip the face
        return

    #denominator used to calculate for barycentric weight incrementation
    denom = (B[0] - C[0]) * (A[1] - C[1]) - (B[1] - C[1]) * (A[0] - C[0])

    #alpha and beta for the first (top left) point in bounding box
    x0,y0=min_x,min_y
    alpha_0 = ((B[0] - C[0]) * (y0 - C[1]) - (B[1] - C[1]) * (x0 - C[0])) / denom
    beta_0 = ((C[0] - A[0]) * (y0 - A[1]) - (C[1] - A[1]) * (x0 - A[0])) / denom

    #change in barycentric weights per change in x/y
    dalpha_x = -(B[1] - C[1]) / denom
    dalpha_y = (B[0] - C[0]) / denom
    dbeta_x = -(C[1] - A[1]) / denom
    dbeta_y = (C[0] - A[0]) / denom

    #diffuse lighting from all 3 average vertices calculated ahead of time
    d1 = max(n1.dot(LIGHT_VECTOR), 0)
    d2 = max(n2.dot(LIGHT_VECTOR), 0)
    d3 = max(n3.dot(LIGHT_VECTOR), 0)

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):

            #calculates 2d barycentric weights using previously found change values
            alpha = alpha_0 + (x - min_x) * dalpha_x + (y - min_y) * dalpha_y
            beta = beta_0 + (x - min_x) * dbeta_x + (y - min_y) * dbeta_y
            gamma = 1 - alpha - beta

            if texture != 0:
                tc = alpha * texturecoords[0] + beta * texturecoords[1] + gamma * texturecoords[2]

                i = int(round(tc[1])) - 1
                j = int(round(tc[0])) - 1
                i = max(0, min(texture.shape[0] - 1, i))
                j = max(0, min(texture.shape[1] - 1, j))

                base = np.asarray(texture[i, j], dtype=np.float64)


            if alpha>=0 and beta>=0 and gamma>=0: #if the current point is in the triangle, continue
                surface_point = alpha * coords_3d[0] + beta * coords_3d[1] + gamma * coords_3d[2]

                if ZBUFF: #zbuffer barely works
                    Z = surface_point[2]
                    if Z >= zbuffer[y, x]:
                        continue
                    zbuffer[y, x] = Z

                # finds normal of point based off of how far it is from vertices
                interpolated_normal = alpha * n1 + beta * n2 + gamma * n3
                interpolated_normal /= np.sqrt(np.dot(interpolated_normal, interpolated_normal))

                #direction from point to light
                light_dir = LIGHT_POS - surface_point
                light_dir /= np.sqrt(np.dot(light_dir, light_dir))

                #direction from point to camera
                view_dir = CAMERA_POS - surface_point
                view_dir /= np.sqrt(np.dot(view_dir, view_dir))

                #The diffuse lighting
                #for some reason using hte interpolated normal just makes it not work
                diffuse = (
                        d1 * alpha +
                        d2 * beta +
                        d3 * gamma
                )

                #what was bui tuong phong on about??
                #calculates unit reflection direction vector
                reflect_dir = 2.0 * np.dot(interpolated_normal, light_dir) * interpolated_normal - light_dir
                reflect_dir /= np.sqrt(np.dot(reflect_dir, reflect_dir))

                #cos of angle between direction of reflection and direction to camera
                spec_angle = max(0.0, np.dot(reflect_dir, view_dir))
                specular = REFLECTIVITY_CONSTANT * (spec_angle ** PHONG_EXPONENT)

                intensity = AMBIENT_INTENSITY + diffuse *255 + specular*255

                #gamma correction
                intensity = 1 * (intensity / 255.0) ** GAMMA

                #caps intensity just in case
                intensity = min(intensity, 1.0)

                color = np.array([base[2]*intensity,base[1]*intensity,base[0]*intensity])
                view[y, x] = color.astype(np.uint8)
    # print(colors)



@numba.njit()
def rasterize_phong(coords, view, zbuffer, av_normals, coords_3d, normal,color = (255,255,255), texturecoords=None):
    base = np.array(color, dtype=np.float64)
    red,green,blue = base
    A, B, C = coords.astype(np.float64)
    n1, n2, n3 = [np.asarray(n, dtype=np.float64) for n in av_normals]
    coords_3d = coords_3d.astype(np.float64)
    base = np.array(color,dtype=np.float64)
    #The 3d coordinated
    x1, y1, z1 = coords_3d[0]
    x2, y2, z2 = coords_3d[1]
    x3, y3, z3 = coords_3d[2]

    #Used for the bounding box
    min_x = max(int(min(A[0], B[0], C[0])), 0)
    max_x = min(int(max(A[0], B[0], C[0])) + 1, view.shape[1])
    min_y = max(int(min(A[1], B[1], C[1])), 0)
    max_y = min(int(max(A[1], B[1], C[1])) + 1, view.shape[0])

    #area of the triangle face
    v1, v2 = A - B, C - B
    total_area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

    if total_area == 0: #if triangle is too small, just skip the face
        return

    #denominator used to calculate for barycentric weight incrementation
    denom = (B[0] - C[0]) * (A[1] - C[1]) - (B[1] - C[1]) * (A[0] - C[0])

    #alpha and beta for the first (top left) point in bounding box
    x0,y0=min_x,min_y
    alpha_0 = ((B[0] - C[0]) * (y0 - C[1]) - (B[1] - C[1]) * (x0 - C[0])) / denom
    beta_0 = ((C[0] - A[0]) * (y0 - A[1]) - (C[1] - A[1]) * (x0 - A[0])) / denom

    #change in barycentric weights per change in x/y
    dalpha_x = -(B[1] - C[1]) / denom
    dalpha_y = (B[0] - C[0]) / denom
    dbeta_x = -(C[1] - A[1]) / denom
    dbeta_y = (C[0] - A[0]) / denom

    #diffuse lighting from all 3 average vertices calculated ahead of time
    d1 = max(n1.dot(LIGHT_VECTOR), 0)
    d2 = max(n2.dot(LIGHT_VECTOR), 0)
    d3 = max(n3.dot(LIGHT_VECTOR), 0)

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):

            #calculates 2d barycentric weights using previously found change values
            alpha = alpha_0 + (x - min_x) * dalpha_x + (y - min_y) * dalpha_y
            beta = beta_0 + (x - min_x) * dbeta_x + (y - min_y) * dbeta_y
            gamma = 1 - alpha - beta



            if alpha>=0 and beta>=0 and gamma>=0: #if the current point is in the triangle, continue
                surface_point = alpha * coords_3d[0] + beta * coords_3d[1] + gamma * coords_3d[2]

                if ZBUFF: #zbuffer barely works
                    Z = surface_point[2]
                    if Z >= zbuffer[y, x]:
                        continue
                    zbuffer[y, x] = Z

                # finds normal of point based off of how far it is from vertices
                interpolated_normal = alpha * n1 + beta * n2 + gamma * n3
                interpolated_normal /= np.sqrt(np.dot(interpolated_normal, interpolated_normal))

                #direction from point to light
                light_dir = LIGHT_POS - surface_point
                light_dir /= np.sqrt(np.dot(light_dir, light_dir))

                #direction from point to camera
                view_dir = CAMERA_POS - surface_point
                view_dir /= np.sqrt(np.dot(view_dir, view_dir))

                #The diffuse lighting
                #for some reason using hte interpolated normal just makes it not work
                diffuse = (
                        d1 * alpha +
                        d2 * beta +
                        d3 * gamma
                )

                #what was bui tuong phong on about??
                #calculates unit reflection direction vector
                reflect_dir = 2.0 * np.dot(interpolated_normal, light_dir) * interpolated_normal - light_dir
                reflect_dir /= np.sqrt(np.dot(reflect_dir, reflect_dir))

                #cos of angle between direction of reflection and direction to camera
                spec_angle = max(0.0, np.dot(reflect_dir, view_dir))
                specular = REFLECTIVITY_CONSTANT * (spec_angle ** PHONG_EXPONENT)

                intensity = AMBIENT_INTENSITY + diffuse *255 + specular*255

                #gamma correction
                intensity = 1 * (intensity / 255.0) ** GAMMA

                #caps intensity just in case
                intensity = min(intensity, 1.0)

                color = np.array([base[2]*intensity,base[1]*intensity,base[0]*intensity])
                view[y, x] = color.astype(np.uint8)
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
        rasterize_phong(coords,view,zbuffer,face.avNorms,face.points,face.normal,(255,255,255))






if __name__ == '__main__':

    c = OBJFile("models/Monkey.obj",reverseNormals=False,loadAverageNorms=True,)#texture="textures/Shambler.png")


    t = time()
    pretransformation()
    c.update2dCoords()
    shader = Lambertian()

    print(f"Prefunctioning: {time() - t} seconds")
    t = time()

    while cv.waitKey(20)&0xff != ord('x'):
        view = np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8)

        if BENCHMARK: now = time()

        t=time()

        display(c)
        cv.imshow("3d Render",view)

        print(f"Display: {time() - t} seconds")

        if BENCHMARK: print(f"Displaying:",time()-now,"seconds"); now = time()

        TransformationLoop()
        c.update2dCoords()

        if BENCHMARK: print("Transforming:",time()-now,"seconds")