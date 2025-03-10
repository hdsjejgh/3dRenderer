from time import sleep
from shapes import *



if __name__ == '__main__':

    c=OBJFile(
        'cube.obj'
    )
    c.shiftCoords('z',0)
    c.update2dCoords()

    #c.display()
    while True: #main loop
        c.display()
        #c.rotateCoords('x',6)
        c.rotateCoords('y', -math.pi)
        #c.rotateCoords('z', math.e)
        c.update2dCoords()
        sleep(0.05)

