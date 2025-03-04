import random
from abc import ABC, abstractmethod
import math
import os
from time import sleep

FOV = 15
symbols = tuple(i+str(idx)+"\033[0m" for idx,i in enumerate(("\033[0;31m", "\033[0;32m", "\033[0;34m", "\033[1;33m", "\033[0;36m", "\033[1;35m"))) #what symbol to use for each face (the weird symbols are ansi color codes to add colors)

def sin(deg: int|float) -> float: #sine function for degrees
    return math.sin((deg*math.pi)/180)
def cos(deg: int|float) -> float: #cosine function for degrees
    return math.cos((deg*math.pi)/180)

class Shape(ABC): #base class for all shapes (will add more shapes later)
    def __init__(self, coords: list[list[int|float]], faces: list[list[int|float]], centerCoords=None,symbols=symbols):
        self.coords = coords
        self.faces = faces
        self.TwoDimensionalCoords = [(i[0]*FOV/(i[2]+FOV),i[1]*FOV/(i[2]+FOV)) for i in coords]
        self.avZ = [sum(coords[ii][-1] for ii in i)/len(i) for i in faces] #average Z value of all faces (used for seeing which face to display on top)

        self.center = centerCoords if centerCoords is not None else self.cc() #center coordinates of rotation
    @abstractmethod
    def cc(self): #function for center coordinate, different for each shape
        pass
    @abstractmethod
    def display(self): #function for displayed shape, different for each shape
        pass

    def update2dCoords(self): #updates 2d coords and avZ for after rotation
        self.TwoDimensionalCoords = [(i[0] * FOV / (i[2] + FOV), i[1] * FOV / (i[2] + FOV)) for i in self.coords]
        self.avZ = [sum(self.coords[ii][-1] for ii in i) / len(i) for i in self.faces]
    def rotateCoords(self,axis: str,  angle:int|float,  center=None,  coords=None): #rotates coordinates
        if coords is None: #if no coords given, use object's coords
            coords = self.coords
        if center is None: #if no center given, use object's center
            center = self.center
        returnCoords = coords.copy() #edits coords in here so you dont change coords based on rotated coords
        axis = axis.lower()
        assert axis in ('x','y','z'), "Invalid axis, Axis must be 'x','y', or 'z'"
        indices = {0,1,2}
        indices.remove({'x':0,'y':1,'z':2}[axis]) #i was gonna do something with the numbers but I forgo
        for idx,triple in enumerate(returnCoords):
            x = triple[0]
            y = triple[1]
            z = triple[2]
            c1 = center[0]
            c2 = center[1]
            c3 = center[2]

            if 0 not in indices:
                #Rotation Matrix about X axis
                #
                #|     1      0          0      |
                #|     0  cos(angle) -sin(angle)|
                #|     0  sin(angle) cos(angle) |
                returnCoords[idx][1] = (y-c2)*cos(angle) + (z-c3)*-sin(angle) + c2 #y
                returnCoords[idx][2] = (y-c2)*sin(angle) + (z-c3)*cos(angle) + c3 #z
            if 1 not in indices:
                #Rotation Matrix about Y axis
                #
                #| cos(angle) 0 sin(angle) |
                #|     0      1     0      |
                #|-sin(angle) 0 cos(angle) |
                returnCoords[idx][0] = (x-c1)*cos(angle) + (z-c3)*sin(angle) + c1 #x
                returnCoords[idx][2] = -(x-c1)*sin(angle) + (z-c3)*cos(angle) + c3 #z
            if 2 not in indices:
                #Rotation Matrix about Z axis
                #
                #|cos(angle) sin(angle)   0      |
                #|-sin(angle) cos(angle    0      |
                #|     0        0         1      |
                returnCoords[idx][0] = (x-c1)*cos(angle) + (y-c2)*sin(angle) + c1 #x
                returnCoords[idx][1] = -(x-c1)*sin(angle) + (y-c2)*cos(angle) + c2 #y


        self.coords= returnCoords




class Cube(Shape):
    def __init__(self, coords: list[list[int|float]], faces: list[list[int|float]], centerCoords=None,symbols=symbols):
        super().__init__(coords, faces)
    def display(self):
        yScale = 10
        xScale = 15 #x scale has to be slightly larger bc character width is less than height

        #iterates through every pixel on grid
        for y in (i * (3/yScale) for i in range(-yScale, yScale)):
            disp = ""
            for x in (i * (3/xScale) for i in range(-xScale, xScale)):
                front = []
                for idx, face in enumerate(self.faces):
                    #finds if point is in each face by summing area of triangle made by point and each 2 consecutive vertices
                    #if sum of areas = area of quad, point in quad, else its outside of it
                    s = triArea(x, y, self.TwoDimensionalCoords[face[0]][0], self.TwoDimensionalCoords[face[0]][1], #triangle made of point, and first and last vertices
                                  self.TwoDimensionalCoords[face[-1]][0], self.TwoDimensionalCoords[face[-1]][1])
                    for i in range(3): #adds sum for other triangles
                        x1 = x
                        y1 = y
                        x2 = self.TwoDimensionalCoords[face[i]][0]
                        y2 = self.TwoDimensionalCoords[face[i]][1]
                        x3 = self.TwoDimensionalCoords[face[i + 1]][0]
                        y3 = self.TwoDimensionalCoords[face[i + 1]][1]
                        s += triArea(x1, y1, x2, y2, x3, y3)

                    x1, y1 = self.TwoDimensionalCoords[face[0]]
                    x2, y2 = self.TwoDimensionalCoords[face[1]]
                    x3, y3 = self.TwoDimensionalCoords[face[2]]
                    x4, y4 = self.TwoDimensionalCoords[face[3]]
                    area = quadArea(x1, y1, x2, y2, x3, y3, x4, y4)

                    # print(f"({x,y}) {area=} {s=} {idx=}")
                    if round(area, 5) == round(s, 5): #if 2d point in 2d representation of 2d face
                        front.append((idx, self.avZ[idx]))
                if len(front) == 0: #if point isnt on any face, add empty face
                    disp += "  "
                    continue
                front.sort(key=lambda x: x[1])
                disp += symbols[front[0][0]] + ' ' #adds symbol for frontmost face
            #if disp.count(' ') != len(disp):
            print(disp)
    def cc(self): #gets center of cube
        minx,maxx, miny,maxy, minz,maxz = [128,-128]*3
        for coord in self.coords:
            minx = min(minx,coord[0])
            maxx = max(maxx, coord[0])

            miny = min(miny, coord[1])
            maxy = max(maxy, coord[1])

            minz = min(minz, coord[2])
            maxz = max(maxz, coord[2])
        return (minx+maxx)/2,(miny+maxy)/2,(minz+maxz)/2

def triArea(x1,y1,x2,y2,x3,y3): #area of a triangle using 3 coordinate pairs
    return (1 / 2) * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def quadArea(x1,y1,x2,y2,x3,y3,x4,y4): #area of a quadrilateral using 4 coordinate pairs
    return 1/2 * abs((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (x2*y1 + x3*y2 + x4*y3 + x1*y4))


c=Cube( #the actual cube
    coords=[[-2,2,2],[2,2,2],[2,-2,2],[-2,-2,2],[-2,2,6],[2,2,6],[2,-2,6],[-2,-2,6],],
    faces=[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2],[0,1,5,4],[2,3,7,6],]
)
#c.display()
while True: #main loop
    c.display()
    c.rotateCoords('x',6)
    c.rotateCoords('y', -math.pi)
    c.rotateCoords('z', 1.5)
    c.update2dCoords()
    sleep(0.05)
    os.system('cls') #clears terminal
