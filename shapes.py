from abc import ABC, abstractmethod
import math
#from main import FOV
from parameters import *
import numpy as np
from time import time
from multiprocessing import Pool

def sin(deg: int|float) -> float: #sine function for degrees
    return math.sin((deg*math.pi)/180)
def cos(deg: int|float) -> float: #cosine function for degrees
    return math.cos((deg*math.pi)/180)

class Shape(ABC): #base class for all shapes (will add more shapes later)

    class face:
        def __init__(self,outerInstance, indices, id):
            self.id=id
            self.indices = np.array(indices)
            self.outerInstance = outerInstance
            self.z = np.mean(outerInstance.coords[indices, 2])

            points = self.outerInstance.coords[indices]
            denominator = points[:, 2] + FOV
            self.TwoDCoords = np.stack((points[:, 0] * FOV / denominator,points[:, 1] * FOV / denominator), axis=1)
            if BACKFACECULLING:
                p0,p1,p2 = outerInstance.coords[indices[:3]]
                v1 = p0-p1
                v2 = p2-p1
                self.normal = np.cross(v1,v2)
                self.normal /= np.linalg.norm(self.normal)

        def __lt__(self, other):
            return self.z<other.z

        def __to2d__(self):
            pass

        # def update(self):
        #     self.z = np.mean(self.outerInstance.coords[self.indices, 2])
        #     points = self.outerInstance.coords[self.indices]
        #     denominator = points[:, 2] + FOV
        #     self.TwoDCoords = np.stack((points[:, 0] * FOV / denominator, points[:, 1] * FOV / denominator), axis=1)
        #     return self


    def __init__(self, coords: list[list[int|float]], faces: list[list[int|float]], centerCoords=None):
        self.coords = np.array(coords)
        self.faces = [self.face(self,faces[i],i) for i in range(len(faces))]

        self.center = centerCoords if centerCoords is not None else self.cc() #center coordinates of rotation

    def cc(self): #function for center coordinate, different for each shape
        mins = np.min(self.coords, axis=0)
        maxs = np.max(self.coords, axis=0)
        return (mins + maxs) / 2


    def update2dCoords(self): #updates 2d coords and avZ for after rotation
        indices = np.array([face.indices for face in self.faces])
        points = np.array([self.coords[idx] for idx in indices])
        z = np.mean(points[:, :, 2], axis=1)
        denominator = points[:, :, 2] + FOV
        TwoDCoords = np.stack((points[:, :, 0] * FOV / denominator, points[:, :, 1] * FOV / denominator), axis=2)

        for i, face in enumerate(self.faces):
            face.z = z[i]
            face.TwoDCoords = TwoDCoords[i]

        self.faces.sort(reverse=True)


    def validFaces(self):
        if not BACKFACECULLING:
            return self.faces
        else:
            return tuple(filter(lambda x: np.dot(x.normal,VIEW_VECTOR)<0, self.faces))

    def rotateCoords(self,axis: str,  angle:int|float): #rotates coordinates

        axis = axis.lower()
        assert axis in ('x','y','z'), "Invalid axis, Axis must be 'x','y', or 'z'"
        c1,c2,c3 = self.center
        sa,ca = sin(angle),cos(angle)

        # Rotation Matrix about X axis
        #
        # |     1      0          0      |
        # |     0  cos(angle) -sin(angle)|
        # |     0  sin(angle) cos(angle) |

        # Rotation Matrix about Y axis
        #
        # | cos(angle) 0 sin(angle) |
        # |     0      1     0      |
        # |-sin(angle) 0 cos(angle) |

        # Rotation Matrix about Z axis
        #
        # |cos(angle) sin(angle)   0      |
        # |-sin(angle) cos(angle    0      |
        # |     0        0         1      |
        if axis == 'x':
            rotMat = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
        elif axis == 'y':
            rotMat = np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])
        else:
            rotMat = np.array([[ca,sa,0],[-sa,ca,0],[0,0,1]])

        shifted = self.coords - self.center
        shifted = shifted @ rotMat.T
        self.coords = shifted+self.center


        if BACKFACECULLING:
            for idx,face in enumerate(self.faces):
                face.normal = face.normal @ rotMat.T



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

        self.coords = np.array(self.coords)
        self.faces = np.array(self.faces)

        #print(self.coords)
        self.faces = [self.face(self,self.faces[i],i) for i in range(len(self.faces))]

        self.center = self.cc()



def mean(arr):
    return sum(arr)/len(arr)