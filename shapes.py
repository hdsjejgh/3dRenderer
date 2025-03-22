from abc import ABC, abstractmethod
import math
#from main import FOV
from parameters import *


def sin(deg: int|float) -> float: #sine function for degrees
    return math.sin((deg*math.pi)/180)
def cos(deg: int|float) -> float: #cosine function for degrees
    return math.cos((deg*math.pi)/180)

class Shape(ABC): #base class for all shapes (will add more shapes later)

    class face:
        def __init__(self,outerInstance, indices, id):
            self.id=id
            self.indices = indices
            self.outerInstance = outerInstance
            self.z = mean([outerInstance.coords[i][2] for i in indices])
            self.TwoDCoords = tuple((self.outerInstance.coords[i][0]*FOV/(self.outerInstance.coords[i][2]+FOV),self.outerInstance.coords[i][1]*FOV/(self.outerInstance.coords[i][2]+FOV)) for i in indices)
            if BACKFACECULLING:
                self.normal = self.__crossprod__(self.__pointToVec__(self.outerInstance.coords[indices[0]], self.outerInstance.coords[indices[1]]), self.__pointToVec__(self.outerInstance.coords[indices[1]], self.outerInstance.coords[indices[2]]))

        def __lt__(self, other):
            return self.z<other.z

        def __to2d__(self):
            pass

        def update(self):
            self.z = mean([self.outerInstance.coords[i][2] for i in self.indices])
            self.TwoDCoords = tuple((self.outerInstance.coords[i][0] * FOV / (self.outerInstance.coords[i][2] + FOV),self.outerInstance.coords[i][1] * FOV / (self.outerInstance.coords[i][2] + FOV)) for i in self.indices)

        def __crossprod__(self,v1,v2):
            vector = (v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0])
            s = math.sqrt(sum(map(lambda x: x**2, vector)))
            return tuple(map(lambda x:-x/s, vector))

        def __pointToVec__(self,p1,p2):
            return tuple(p1[i]-p2[i] for i in range(3))


    def __init__(self, coords: list[list[int|float]], faces: list[list[int|float]], centerCoords=None):
        self.coords = coords
        self.faces = [self.face(self,faces[i],i) for i in range(len(faces))]

        self.center = centerCoords if centerCoords is not None else self.cc() #center coordinates of rotation
    @abstractmethod
    def cc(self): #function for center coordinate, different for each shape
        pass
    def update2dCoords(self): #updates 2d coords and avZ for after rotation
        for face in self.faces:
            face.update()
        self.faces.sort(reverse=True)


    def validFaces(self):
        if not BACKFACECULLING:
            return self.faces
        else:
            return tuple(filter(lambda x: x.normal[2]<=0, self.faces))

    def __dotProd__(self,v1,v2):
        assert len(v1)==len(v2), "Invalid vector lengths"

        return sum(v1[i]+v2[i] for i in range(len(v1)))

    def rotateCoords(self,axis: str,  angle:int|float): #rotates coordinates
        coords = self.coords
        axis = axis.lower()
        assert axis in ('x','y','z'), "Invalid axis, Axis must be 'x','y', or 'z'"
        indices = {0,1,2}
        indices.remove({'x':0,'y':1,'z':2}[axis]) #i was gonna do something with the numbers but I forgo
        c1 = self.center[0]
        c2 = self.center[1]
        c3 = self.center[2]
        sa,ca = sin(angle),cos(angle)
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
                t[1] = (y-c2)*ca + (z-c3)*-sa + c2 #y
                t[2] = (y-c2)*sa + (z-c3)*ca + c3 #z
            elif 1 not in indices:
                #Rotation Matrix about Y axis
                #
                #| cos(angle) 0 sin(angle) |
                #|     0      1     0      |
                #|-sin(angle) 0 cos(angle) |
                t[0] = (x-c1)*ca + (z-c3)*sa + c1 #x
                t[2] = -(x-c1)*sa + (z-c3)*ca + c3 #z
            elif 2 not in indices:
                #Rotation Matrix about Z axis
                #
                #|cos(angle) sin(angle)   0      |
                #|-sin(angle) cos(angle    0      |
                #|     0        0         1      |
                t[0] = (x-c1)*ca + (y-c2)*sa + c1 #x
                t[1] = -(x-c1)*sa + (y-c2)*ca + c2 #y
            coords[idx] = t

        if BACKFACECULLING:
            for idx,face in enumerate(self.faces):
                t=list(face.normal)
                x,y,z = t
                if 0 not in indices:
                    t[1] = (y) * ca + (z) * -sa  # y
                    t[2] = (y) * sa + (z) * ca  # z
                elif 1 not in indices:
                    t[0] = (x) * ca + (z) * sa  # x
                    t[2] = -(x) * sa + (z) * ca  # z
                elif 2 not in indices:
                    t[0] = (x) * ca + (y) * sa  # x
                    t[1] = -(x) * sa + (y) * ca  # y
                s = math.sqrt(sum(map(lambda x: x**2, t)))
                self.faces[idx].normal=tuple(map(lambda x: x / s, t))


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

        assert len(coords) == 8, "Invalid vertex count"
        assert len(faces) == 6, "Invalid face count"

        super().__init__(coords, faces)

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
                    #print(line)
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
                    exit()
        #print(self.coords)
        self.faces = [self.face(self,self.faces[i],i) for i in range(len(self.faces))]
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

def mean(arr):
    return sum(arr)/len(arr)