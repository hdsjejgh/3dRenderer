from abc import ABC, abstractmethod
import math
import os
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
from time import sleep

FOV = 30
symbols = tuple(i+str(idx)+"\033[0m" for idx,i in enumerate(("\033[0;31m", "\033[0;32m", "\033[0;34m", "\033[1;33m", "\033[0;36m", "\033[1;35m")))

def sin(deg: int|float) -> float:
    return math.sin((deg*math.pi)/180)
def cos(deg: int|float) -> float:
    return math.cos((deg*math.pi)/180)

class Shape(ABC):
    def __init__(self, coords: list[list[int|float]], faces: list[list[int|float]], centerCoords=None,symbols=symbols):
        self.coords = coords
        self.faces = faces
        self.TwoDimensionalCoords = [(i[0]*FOV/(i[2]+FOV),i[1]*FOV/(i[2]+FOV)) for i in coords]
        self.avZ = [sum(coords[ii][-1] for ii in i)/len(i) for i in faces]

        self.center = centerCoords if centerCoords is not None else self.cc()
    @abstractmethod
    def cc(self):
        pass
    @abstractmethod
    def display(self):
        pass

    def update2dCoords(self):
        self.TwoDimensionalCoords = [(i[0] * FOV / (i[2] + FOV), i[1] * FOV / (i[2] + FOV)) for i in self.coords]
        self.avZ = [sum(self.coords[ii][-1] for ii in i) / len(i) for i in self.faces]
    def rotateCoords(self,axis: str,  angle:int|float,  center=None,  coords=None):
        if coords is None:
            coords = self.coords
        if center is None:
            center = self.center
        returnCoords = coords.copy()
        axis = axis.lower()
        assert axis in ('x','y','z'), "Invalid axis, Axis must be 'x','y', or 'z'"
        indices = {0,1,2}
        indices.remove({'x':0,'y':1,'z':2}[axis])
        for idx,triple in enumerate(returnCoords):
            if 1 not in indices:
                x=triple[0]
                z=triple[2]
                c1 = center[0]
                c2=center[2]
                #print(f"{x=} {z=} {c1=} {c2=}")
                returnCoords[idx][0] = (x-c1)*cos(angle) + (z-c2)*sin(angle) + c1 #x
                returnCoords[idx][2] = -(x-c1)*sin(angle) + (z-c2)*cos(angle) + c2 #z
        self.coords= returnCoords




class Cube(Shape):
    def __init__(self, coords: list[list[int|float]], faces: list[list[int|float]], centerCoords=None,symbols=symbols):
        super().__init__(coords, faces)
    def display(self):
        for y in (i * 0.3 for i in range(-10, 10)):
            disp = ""
            for x in (i * 0.3 for i in range(-10, 10)):
                front = []
                for idx, face in enumerate(self.faces):
                    s = triArea(x, y, self.TwoDimensionalCoords[face[0]][0], self.TwoDimensionalCoords[face[0]][1],
                                  self.TwoDimensionalCoords[face[-1]][0], self.TwoDimensionalCoords[face[-1]][1])
                    for i in range(3):
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
                    if round(area, 5) == round(s, 5):
                        front.append((idx, self.avZ[idx]))
                if len(front) == 0:
                    disp += "  "
                    continue
                front.sort(key=lambda x: x[1])
                disp += symbols[front[0][0]] + ' '
            if disp.count(' ') != len(disp):
                print(disp)
    def cc(self):
        minx,maxx, miny,maxy, minz,maxz = [128,-128]*3
        for coord in self.coords:
            minx = min(minx,coord[0])
            maxx = max(maxx, coord[0])

            miny = min(miny, coord[1])
            maxy = max(maxy, coord[1])

            minz = min(minz, coord[2])
            maxz = max(maxz, coord[2])
        return (minx+maxx)/2,(miny+maxy)/2,(minz+maxz)/2

def triArea(x1,y1,x2,y2,x3,y3):
    return (1 / 2) * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def quadArea(x1,y1,x2,y2,x3,y3,x4,y4):
    return 1/2 * abs((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (x2*y1 + x3*y2 + x4*y3 + x1*y4))


c=Cube(
    coords=[[-2,2,2],[2,2,2],[2,-2,2],[-2,-2,2],[-2,2,6],[2,2,6],[2,-2,6],[-2,-2,6],],
    faces=[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2],[0,1,5,4],[2,3,7,6],]
)
#c.display()
while True:
    print()
    print()
    print()
    c.display()
    c.rotateCoords('y',5)
    c.update2dCoords()
    sleep(0.5)
