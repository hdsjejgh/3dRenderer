from time import sleep
from shapes import *
import pygame
import random
from parameters import *




c=Cube(
        coords=[[-2,2,2],[2,2,2],[2,-2,2],[-2,-2,2],[-2,2,6],[2,2,6],[2,-2,6],[-2,-2,6],],
         faces=[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2],[0,1,5,4],[2,3,7,6],]
    )
#c=OBJFile("Shambler.obj")
c.scaleCoords(-20.75)
c.update2dCoords()

def TransformationLoop():
    #c.rotateCoords('x', 6)
    c.rotateCoords('y', -6)
    #c.rotateCoords('z', math.e)
    c.update2dCoords()
    sleep(0.0)

def display(shape):
    def center(x):
        x=list(x)
        x[0]+=WIDTH/2
        x[1]+=HEIGHT/2
        return x
    for face in shape.faces:
        pygame.draw.polygon(screen, tuple(random.randint(0, 255) for i in range(3)), list(map(center,face.TwoDCoords)))
    pygame.display.flip()

if __name__ == '__main__':

    pygame.init()
    screen = pygame.display.set_mode((WIDTH,HEIGHT))


    running = True
    clock = pygame.time.Clock()

    while running:

        for event in pygame.event.get():
            if event == pygame.QUIT:
                running=False


        screen.fill('black')
        TransformationLoop()
        c.update2dCoords()


        display(c)
        clock.tick(FPS)


    pygame.quit()



