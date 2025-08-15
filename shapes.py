from abc import ABC, abstractmethod
import math
#from main import FOV
from parameters import *
import numpy as np
from time import time
from multiprocessing import Pool
import numba
from collections import defaultdict

def sin(deg: int|float) -> float: #sine function for degrees
    return math.sin((deg*math.pi)/180)
def cos(deg: int|float) -> float: #cosine function for degrees
    return math.cos((deg*math.pi)/180)

class OBJFile():
    def __init__(self,filepath, reverseNormals=False, loadAverageNorms=False, *args, **kwargs):
        self.coords = []
        self.faces = []

        self.reverseNormals = reverseNormals
        t = time()
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
        print(f"Loading: {time()-t} seconds")
        t = time()
        self.coords = np.array(self.coords)
        self.faces = np.array(self.faces)

        #print(self.coords)
        self.faces = [self.face(self,self.faces[i],i) for i in range(len(self.faces))]

        print(f"Faces: {time() - t} seconds")
        t = time()

        self.mapping = defaultdict(list)
        for i,face in enumerate(self.faces):
            for v in face.indices:
                self.mapping[v].append(i)

        print(f"Mapping: {time() - t} seconds")
        t = time()

        if loadAverageNorms:

            self.get_borders()
        self.center = self.cc()

        print(f"Borders: {time() - t} seconds")
        t = time()

    class face:
        def __init__(self,outerInstance, indices, id):
            reverse = outerInstance.reverseNormals
            self.id=id
            self.indices = np.array(indices)
            c = -1 if outerInstance.reverseNormals else 1
            self.outerInstance = outerInstance
            self.z = np.mean(outerInstance.coords[indices, 2])
            self.points = self.outerInstance.coords[indices]
            denominator = self.points[:, 2] + FOV
            self.TwoDCoords = np.stack((self.points[:, 0] * FOV / denominator,self.points[:, 1] * FOV / denominator), axis=1)
            p0,p1,p2 = outerInstance.coords[indices[:3]]
            v1 = p0-p1
            v2 = p2-p1
            self.normal = np.cross(v1,v2)
            if reverse:
                self.normal*=-1
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



    def cc(self): #function for center coordinate, different for each shape
        mins = np.min(self.coords, axis=0)
        maxs = np.max(self.coords, axis=0)
        return (mins + maxs) / 2

    def get_borders(self):
        for face in self.faces:
            face.bordering = [self.mapping[i] for i in face.indices]
            face.avNorms = [np.mean([self.faces[id].normal for id in vertex],axis=0)/np.linalg.norm(np.mean([self.faces[id].normal for id in vertex],axis=0)) for vertex in face.bordering]

    def update2dCoords(self): #updates 2d coords and avZ for after rotation
        indices = np.array([face.indices for face in self.faces])
        points = np.array([self.coords[idx] for idx in indices])
        z = np.mean(points[:, :, 2], axis=1)
        denominator = points[:, :, 2] + FOV
        TwoDCoords = np.stack((points[:, :, 0] * FOV / denominator, points[:, :, 1] * FOV / denominator), axis=2)

        for i, face in enumerate(self.faces):
            face.z = z[i]
            face.TwoDCoords = TwoDCoords[i]
            face.points = self.coords[face.indices]

        self.faces.sort(reverse=True)


    def validFaces(self):
        #for some reason, backface culling does not want to work well for some faces
        return tuple(filter(lambda x: np.dot(x.normal,VIEW_VECTOR)<1e-2, self.faces))

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

        for idx,face in enumerate(self.faces):
            face.normal = face.normal @ rotMat.T
            try:
                face.avNorms = face.avNorms @ rotMat.T
            except:
                pass

    def matrixTransform(self,matrix): #rotates coordinates
        arr = np.array(matrix)



        shifted = self.coords - self.center
        shifted = shifted @ arr.T
        self.coords = shifted+self.center

        for idx,face in enumerate(self.faces):
            face.normal = face.normal @ arr.T
            try:
                face.avNorms = face.avNorms @ arr.T
            except:
                pass


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




@numba.njit()
def rasterize(coords, color, view):
    A, B, C = coords

    #Used for the bounding box
    min_x = max(int(min(A[0], B[0], C[0])), 0)
    max_x = min(int(max(A[0], B[0], C[0])) + 1, view.shape[1])
    min_y = max(int(min(A[1], B[1], C[1])), 0)
    max_y = min(int(max(A[1], B[1], C[1])) + 1, view.shape[0])

    #area of triangle
    area = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
    if area == 0:
        return

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            #barycentric coordinates are the goat
            w0 = (B[0] - A[0]) * (y - A[1]) - (B[1] - A[1]) * (x - A[0])
            w1 = (C[0] - B[0]) * (y - B[1]) - (C[1] - B[1]) * (x - B[0])
            w2 = (A[0] - C[0]) * (y - C[1]) - (A[1] - C[1]) * (x - C[0])

            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):


                view[y, x] = color


def mean(arr):
    return sum(arr)/len(arr)