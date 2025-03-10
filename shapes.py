from abc import ABC, abstractmethod
import math
import os
from decimal import Decimal, getcontext

getcontext().prec = 30

FOV = 15
symbols = tuple(i+str(idx)+"\033[0m" for idx,i in enumerate(("\033[0;31m", "\033[0;32m", "\033[0;34m", "\033[1;33m", "\033[0;36m", "\033[1;35m"))) #what symbol to use for each face (the weird symbols are ansi color codes to add colors)
X_SCALE = 15
Y_SCALE = 10

def sin(deg: int|float) -> float: #sine function for degrees
    return math.sin((deg*math.pi)/180)
def cos(deg: int|float) -> float: #cosine function for degrees
    return math.cos((deg*math.pi)/180)

class Shape(ABC): #base class for all shapes (will add more shapes later)
    def __init__(self, coords: list[list[int|float]], faces: list[list[int|float]], centerCoords=None):
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



    def rotateCoords(self,axis: str,  angle:int|float): #rotates coordinates
        coords = self.coords
        axis = axis.lower()
        assert axis in ('x','y','z'), "Invalid axis, Axis must be 'x','y', or 'z'"
        indices = {0,1,2}
        indices.remove({'x':0,'y':1,'z':2}[axis]) #i was gonna do something with the numbers but I forgo
        c1 = self.center[0]
        c2 = self.center[1]
        c3 = self.center[2]
        for idx,triple in enumerate(coords):
            x = triple[0]
            y = triple[1]
            z = triple[2]
            t = triple.copy()

            if 0 not in indices:
                #Rotation Matrix about X axis
                #
                #|     1      0          0      |
                #|     0  cos(angle) -sin(angle)|
                #|     0  sin(angle) cos(angle) |
                t[1] = (y-c2)*cos(angle) + (z-c3)*-sin(angle) + c2 #y
                t[2] = (y-c2)*sin(angle) + (z-c3)*cos(angle) + c3 #z
            if 1 not in indices:
                #Rotation Matrix about Y axis
                #
                #| cos(angle) 0 sin(angle) |
                #|     0      1     0      |
                #|-sin(angle) 0 cos(angle) |
                t[0] = (x-c1)*cos(angle) + (z-c3)*sin(angle) + c1 #x
                t[2] = -(x-c1)*sin(angle) + (z-c3)*cos(angle) + c3 #z
            if 2 not in indices:
                #Rotation Matrix about Z axis
                #
                #|cos(angle) sin(angle)   0      |
                #|-sin(angle) cos(angle    0      |
                #|     0        0         1      |
                t[0] = (x-c1)*cos(angle) + (y-c2)*sin(angle) + c1 #x
                t[1] = -(x-c1)*sin(angle) + (y-c2)*cos(angle) + c2 #y
            coords[idx] = t

        self.coords= coords

    def shiftCoords(self, axis: str, amount:int|float,coords=None):
        if coords is None: #if no coords given, use object's coords
            coords = self.coords.copy()

        axis = axis.lower()
        assert axis in ('x', 'y', 'z'), "Invalid axis, Axis must be 'x','y', or 'z'"
        axis = {'x':0,'y':1,'z':2}[axis]
        for idx,triple in enumerate(coords):
            coords[idx][axis] = triple[axis]+amount
        self.coords = coords
        self.center = self.cc()

    def scaleCoords(self, amount:int|float):
        coords = self.coords
        c1 = self.center[0]
        c2 = self.center[1]
        c3 = self.center[2]
        for idx, triple in enumerate(coords):
            x = triple[0]
            y = triple[1]
            z = triple[2]
            t = triple.copy()
            t[0] = amount*(t[0]-c1)+c1
            t[1] = amount * (t[1] - c2) + c2
            t[2] = amount * (t[2] - c3) + c3
            coords[idx] = t
        self.coords=coords





class Cube(Shape):
    def __init__(self, coords: list[list[int|float]], faces: list[list[int|float]], centerCoords=None,):
        super().__init__(coords, faces)
    def display(self):
        os.system('cls') #clears terminal
        #iterates through every pixel on grid
        backbuffer = ""
        for y in (i * (3/Y_SCALE) for i in range(-Y_SCALE, Y_SCALE)):

            for x in (i * (3/X_SCALE) for i in range(-X_SCALE, X_SCALE)):
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
                    backbuffer += "  "
                    continue
                frontface = front[1]
                for face in front:
                    if face[1]<frontface[1]:
                        frontface=face
                backbuffer += symbols[frontface[0]] + ' ' #adds symbol for frontmost face
            #if disp.count(' ') != len(disp):
            backbuffer+='\n'
        print(backbuffer)
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

class OBJFile(Shape):
    def __init__(self,filepath, *args, **kwargs):
        self.coords = []
        self.faces = []

        with open(filepath, "r") as file:
            for line in file:
                try:
                    line = line.split()
                    if len(line)<1:
                        continue
                    print(line)
                    type = line[0]

                    if type == 'v':
                        items = [float(line[i]) for i in range(1, 4)]
                        self.coords.append(items)
                    elif type == 'f':
                        items = line[1:]
                        items = list(map(lambda x: x.split('/'),items))
                        face = []
                        for i in items:
                            face.append(int(i[0])-1)
                        self.faces.append(face)
                except Exception as e:
                    print("ERROr")
                    input(e)
        print(self.coords)
        self.TwoDimensionalCoords = [(i[0] * FOV / (i[2] + FOV), i[1] * FOV / (i[2] + FOV)) for i in self.coords]
        self.avZ = [sum(self.coords[ii][-1] for ii in i) / len(i) for i in self.faces]
        self.center = self.cc()
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

    def display(self):
        os.system('cls') #clears terminal
        #iterates through every pixel on grid
        backbuffer = ""
        for y in (i * (3/Y_SCALE) for i in range(-Y_SCALE, Y_SCALE)):

            for x in (i * (3/X_SCALE) for i in range(-X_SCALE, X_SCALE)):
                front = []
                for idx, face in enumerate(self.faces):
                    #finds if point is in each face by summing area of triangle made by point and each 2 consecutive vertices
                    #if sum of areas = area of quad, point in quad, else its outside of it
                    s = triArea(x, y, self.TwoDimensionalCoords[face[0]][0], self.TwoDimensionalCoords[face[0]][1], #triangle made of point, and first and last vertices
                                  self.TwoDimensionalCoords[face[-1]][0], self.TwoDimensionalCoords[face[-1]][1])
                    for i in range(len(face)-1): #adds sum for other triangles
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
                    area = triArea(x1, y1, x2, y2, x3, y3)

                    # print(f"({x,y}) {area=} {s=} {idx=}")
                    if round(area, 5) == round(s, 5): #if 2d point in 2d representation of 2d face
                        front.append((idx, self.avZ[idx]))
                if len(front) == 0: #if point isnt on any face, add empty face
                    #print(s, area, y)
                    #print(self.TwoDimensionalCoords[face[i]][0],self.TwoDimensionalCoords[face[i]][1], self.TwoDimensionalCoords[face[i+1]][0], self.TwoDimensionalCoords[face[i+1]][1])

                    backbuffer += "  "
                    continue
                frontface = front[1]

                for face in front:
                    if face[1]<frontface[1]:
                        frontface=face
                backbuffer += symbols[frontface[0]%len(symbols)] + ' ' #adds symbol for frontmost face
            #if disp.count(' ') != len(disp):
            backbuffer+='\n'
        print(backbuffer)


def triArea(x1,y1,x2,y2,x3,y3): #area of a triangle using 3 coordinate pairs
    x1 = Decimal(x1)
    x2 = Decimal(x2)
    x3 = Decimal(x3)

    y1 = Decimal(y1)
    y2 = Decimal(y2)
    y3 = Decimal(y3)


    return Decimal('0.5') * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def quadArea(x1,y1,x2,y2,x3,y3,x4,y4): #area of a quadrilateral using 4 coordinate pairs
    return 1/2 * abs((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (x2*y1 + x3*y2 + x4*y3 + x1*y4))
