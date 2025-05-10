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
    #c.rotateCoords('z', 1)
    c.update2dCoords()
    #sleep(0.0)

def pretransformation():
    c.scaleCoords(-2)
    # c.shiftCoords('y', 100)

@numba.njit() #I LOVE NUMBA; IT MADE THE CODE SO MUCH QUICKER AND GOT RID OF ALL THE SILLY NUMPY STUFF ITS SO SIMPLE NOW, I OWE TRAVIS OLIPHANT MY LIFE
def rasterize(coords, color, view):
    A, B, C = coords

    #Used for the bounding box
    min_x = max(int(min(A[0], B[0], C[0])), 0)
    max_x = min(int(max(A[0], B[0], C[0])) + 1, view.shape[1])
    min_y = max(int(min(A[1], B[1], C[1])), 0)
    max_y = min(int(max(A[1], B[1], C[1])) + 1, view.shape[0])

    #area of triangle
    area = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
    if area == 0:
        return

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            #barycentric coordinates are the goat
            w0 = (B[0] - A[0]) * (y - A[1]) - (B[1] - A[1]) * (x - A[0])
            w1 = (C[0] - B[0]) * (y - B[1]) - (C[1] - B[1]) * (x - B[0])
            w2 = (A[0] - C[0]) * (y - C[1]) - (A[1] - C[1]) * (x - C[0])

            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                view[y, x] = color

def display(shape,shader):
    if DISPLAY_MODE == 'rasterizer':
        global view
        view = np.zeros((HEIGHT, WIDTH, 3),dtype=np.uint8)
        centArray = np.array([WIDTH/2,HEIGHT/2])
        # print(len(shape.validFaces()))
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

    c = OBJFile("models/Hellknight.obj")
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
            if BENCHMARK: now=time()
            TransformationLoop()

            c.update2dCoords()

            if BENCHMARK: print("Transforming:",time()-now,"seconds");now = time()
            display(c,shader)
            if BENCHMARK: print("Displaying:",time()-now,"seconds")


        pygame.quit()

    if DISPLAY_MODE == 'rasterizer':
        while cv.waitKey(20)&0xff != ord('x'):
            view = np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8)

            if BENCHMARK: now = time()

            display(c,shader)
            cv.imshow("3d Render",view)

            if BENCHMARK: print(f"Displaying:",time()-now,"seconds"); now = time()

            TransformationLoop()
            c.update2dCoords()

            if BENCHMARK: print("Transforming:",time()-now,"seconds")