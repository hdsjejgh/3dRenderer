from time import sleep
from shapes import *
import pygame
import random

WIDTH=800
HEIGHT=800


c=Cube(
        coords=[[-2,2,2],[2,2,2],[2,-2,2],[-2,-2,2],[-2,2,6],[2,2,6],[2,-2,6],[-2,-2,6],],
         faces=[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2],[0,1,5,4],[2,3,7,6],]
    )
c=OBJFile("Shambler.obj")
c.scaleCoords(-3)
c.update2dCoords()

def TransformationLoop():
    #c.rotateCoords('x', 6)
    c.rotateCoords('y', -6)
    #c.rotateCoords('z', math.e)
    c.update2dCoords()
    sleep(0.0)



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


        for face in c.faces[::-1]:
            coords = []
            for point in face:
                coords.append(tuple(c.TwoDimensionalCoords[point][i]+(WIDTH,HEIGHT)[i]/2 for i in range(2)))
            print(c.TwoDimensionalCoords)
            pygame.draw.polygon(screen, tuple(random.randint(0,255) for i in range(3)), coords)
        pygame.display.flip()
        clock.tick(60)


    pygame.quit()



