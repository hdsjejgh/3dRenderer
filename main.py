from time import sleep,time
import numpy as np
from shapes import *
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from parameters import *
from shaders import *
import cv2 as cv

# c=Cube(
#         coords=[[-2,2,2],[2,2,2],[2,-2,2],[-2,-2,2],[-2,2,6],[2,2,6],[2,-2,6],[-2,-2,6],],
#          faces=[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2],[0,1,5,4],[2,3,7,6],]
#     )


def TransformationLoop():
    #c.rotateCoords('x', 4)
    c.rotateCoords('y', -1)
    #c.rotateCoords('z', 1)
    c.update2dCoords()
    #sleep(0.0)

def pretransformation():
    c.scaleCoords(-3)
    # c.shiftCoords('y', 100)

def rasterize(coords,color,view):
    # print(color)
    def cross2d(u, v):
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    A,B,C = coords
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    mins = mins.astype(int)
    maxs = maxs.astype(int)+1
    mins = np.clip(mins,0,800)
    maxs = np.clip(maxs, 0, 800)
    dimensions = maxs-mins
    if np.any(dimensions<=0):
        return
    # print(dimensions)
    # print(mins)
    # print(maxs)
    xl = np.arange(mins[0], maxs[0])
    yl = np.arange(mins[1], maxs[1])
    xs,ys = np.meshgrid(xl,yl)
    grid = np.stack((xs,ys),axis=-1)
    grid = grid.reshape(-1,2)
    grid = grid.astype(float)
    # print(grid)

    def cross_prod_2d(points,vec):
        return points[:,0]*vec[1] - points[:,1]*vec[0]

    ar = (B-A)[0] * (C-A)[1] - (B-A)[1] * (C-A)[0]
    if ar==0:
        return
    v0 = cross_prod_2d(grid-A,B-A)
    v1 = cross_prod_2d(grid-B,C-B)
    v2 = cross_prod_2d(grid-C,A-C)

    signs = np.signbit(v0), np.signbit(v1), np.signbit(v2)
    same_sign = (signs[0] == signs[1]) & (signs[1] == signs[2])
    mask = (np.isclose(v0+v1+v2,ar,1e5)) & same_sign
    # slice = view[mins[1]:maxs[1], mins[0]:maxs[0]]

    mask = mask.reshape(dimensions[1],dimensions[0])
    # slice[mask] = color
    view[mins[1]:maxs[1], mins[0]:maxs[0]][mask] = color


def display(shape,shader):
    if DISPLAY_MODE == 'rasterizer':
        global view
        view = np.zeros((HEIGHT, WIDTH, 3),dtype=np.uint8)
        centArray = np.array([WIDTH/2,HEIGHT/2])
        print(len(shape.validFaces()))
        for face in shape.validFaces():
            coords = face.TwoDCoords+centArray
            color = shader(face)
            color = np.clip(color, 0, 255).astype(np.uint8)
            rasterize(coords,color,view)
    if DISPLAY_MODE == 'pygame':
        def center(x):
            x = list(x)
            x[0] += WIDTH / 2
            x[1] += HEIGHT / 2
            return x
        for face in shape.validFaces():
            pygame.draw.polygon(screen, shader(face), list(map(center,face.TwoDCoords)))
        pygame.display.flip()

if __name__ == '__main__':

    c = OBJFile("models/Shambler.obj")
    pretransformation()
    c.update2dCoords()
    shader = Lambertian()

    if DISPLAY_MODE == 'pygame':
        pygame.init()
        screen = pygame.display.set_mode((WIDTH,HEIGHT))
        running = True
        clock = pygame.time.Clock()
        shader = Lambertian()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running=False


            screen.fill('black')
            # now = time()
            TransformationLoop()
            # print(f"Transformation: {time()-now}")
            # now = time()
            c.update2dCoords()
            # print(f"2difying: {time()-now}")
            # now = time()
            display(c,shader)
            # print(f"Displaying: {time()-now}")
            #clock.tick(FPS)


        pygame.quit()

    if DISPLAY_MODE == 'rasterizer':
        while cv.waitKey(20)&0xff != ord('x'):
            view = np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8)
            display(c,shader)
            cv.imshow("3d Render",view)

            TransformationLoop()
            c.update2dCoords()
