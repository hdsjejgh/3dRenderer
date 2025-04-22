from time import sleep,time
from shapes import *
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import random
from parameters import *
from shaders import *

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

def display(shape,shader):
    def center(x):
        x=list(x)
        x[0]+=WIDTH/2
        x[1]+=HEIGHT/2
        return x
    for face in shape.validFaces():
        pygame.draw.polygon(screen, shader(face), list(map(center,face.TwoDCoords)))
    pygame.display.flip()

if __name__ == '__main__':

    c = OBJFile("models/Hellknight.obj")
    c.scaleCoords(-2)
    c.shiftCoords('y', 100)
    c.update2dCoords()


    pygame.init()
    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running=False


        screen.fill('black')
        now = time()
        TransformationLoop()
        print(f"Transformation: {time()-now}")
        now = time()
        c.update2dCoords()
        print(f"2difying: {time()-now}")
        now = time()
        display(c,sideShadow())
        print(f"Displaying: {time()-now}")
        clock.tick(FPS)


    pygame.quit()