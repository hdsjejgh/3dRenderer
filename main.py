from time import sleep
from shapes import *



if __name__ == '__main__':

    c=Cube( #the actual cube
        coords=[[-2,2,2],[2,2,2],[2,-2,2],[-2,-2,2],[-2,2,6],[2,2,6],[2,-2,6],[-2,-2,6],],
        faces=[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2],[0,1,5,4],[2,3,7,6],]
    )
    #c.display()
    while True: #main loop
        c.display()
        c.rotateCoords('x',6)
        c.rotateCoords('y', -math.pi)
        c.rotateCoords('z', math.e)
        c.update2dCoords()
        sleep(0.05)

