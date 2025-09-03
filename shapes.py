from parameters import *
import numpy as np
from time import time
import numba
from PIL import Image
from collections import defaultdict
import math
#-----------------------------------------------#


#Degrees based sin function
def sin(deg: int|float) -> float:
    return math.sin((deg*math.pi)/180)
#Degrees based cos function
def cos(deg: int|float) -> float:
    return math.cos((deg*math.pi)/180)

def rot_matrix(axis,deg):
    axis = axis.lower()
    assert axis in ('x', 'y', 'z'), "Invalid axis, Axis must be 'x','y', or 'z'"

    # Sin and cosine of angle precomputed just for minor speedups
    sa, ca = sin(deg), cos(deg)

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

    # Forms rotational matrix based on the above
    if axis == 'x':
        rotMat = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    elif axis == 'y':
        rotMat = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    elif axis == 'z':
        rotMat = np.array([[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]])

    return rotMat


#Class for loading a .obj 3d model
class OBJFile():
    def __init__(self,filepath, reverseNormals=False,texture=None, *args, **kwargs):

        #Holds 3d coordinates of file
        self.coords = []

        #Holds the variety of face objects (at first just contains the corresponding coordinate ids)
        self.faces = []

        #Holds vertex normals (if they are in the file)
        self.vertexnormals = []
        #Holds vertex normal ids for each face
        self.vnids = []
        #Whether or not the file had pregiven vertex normals
        self.vnloaded = False

        #Whether or not there is a texture given
        self.textured = False if texture is None else True
        if self.textured:
            #Holds texture coordinates
            self.texturecoords = []
            #Loads and converts texture to np array
            self.texture = Image.open(texture)
            self.texture = np.array(self.texture)
            #Height and width of texture
            self.height = self.texture.shape[0]
            self.width = self.texture.shape[1]
            #Contains which texture coordinate each face corresponds to
            self.textureids = []

        #Whether or not to flip normal vectors
        self.reverseNormals = reverseNormals

        #Used to time variety of things
        t = time()


        with open(filepath, "r") as file:
            for line in file:
                line = line.split()
                #If line is empty, continue
                if len(line)<1: continue

                #What kind of data is given in the line (v,f,vn,vt)
                type = line[0]

                #Vertex point given
                if type == 'v':
                    items = [float(line[i]) for i in range(1, 4)]
                    self.coords.append(items)

                #Face given
                elif type == 'f':

                    #Face data contains data for the 3 coordinates individually
                    items = line[1:]
                    #Separates the individual vertex data
                    items = list(map(lambda x: x.split('/'),items))

                    #Data for coordinates
                    face = []
                    #Data for vertex normals
                    vn = []
                    #Data for texture coordinates
                    tc = []

                    #goes over all 3 vertices
                    for i in items:
                        #0 indexes coordinates as well
                        face.append(int(i[0])-1)
                        vn.append(int(i[2])-1)
                        if self.textured: tc.append(int(i[1]) - 1)

                    self.faces.append(face)
                    self.vnids.append(vn)
                    if self.textured: self.textureids.append(tc)

                #Texture coordinate given
                elif type == 'vt' and self.textured:
                    items = [float(line[i])*[0,self.width,self.height][i] for i in range(1, 3)]
                    #changes <0,0> to be the top left instead of bottom left by reflecting then shifting
                    items[-1]=self.height-items[-1]
                    self.texturecoords.append(items)

                #Vertex normal given
                elif type == 'vn':
                    #Confirms vertex normals were included
                    self.vnloaded = True

                    items = [float(line[i]) for i in range(1, 4)]
                    self.vertexnormals.append(items)

        #Time taken to load file
        print(f"Loading: {time()-t} seconds")
        t = time()

        #Converts everything to a numpy array
        self.coords = np.array(self.coords)
        self.vertexnormals = np.array(self.vertexnormals)
        self.faces = np.array(self.faces)
        if self.textured:
            self.texturecoords = np.array(self.texturecoords)


        #Handles all 4 face information scenarios:
        #no vn, yes textures;  no vn, no textures;  yes vn, yes textures;  yes vn, no textures

        #If no vertex normals are provided, none will be provided to the faces, and they will have to calculate them manually
        if not self.vnloaded:
            if self.textured:
                self.faces = [self.face(self,self.faces[i],i,textureids = self.textureids[i]) for i in range(len(self.faces))]
            else:
                self.faces = [self.face(self, self.faces[i], i) for i in range(len(self.faces))]

        #If vertex normals are provided, faces won't have to calculate them (quicker and more accurate)
        else:
            if self.textured:
                self.faces = [self.face(self,self.faces[i],i,textureids = self.textureids[i],vn=self.vnids[i]) for i in range(len(self.faces))]
            else:
                self.faces = [self.face(self, self.faces[i], i,vn=self.vnids[i]) for i in range(len(self.faces))]

        #Time taken to make faces
        print(f"Faces: {time() - t} seconds")
        t = time()

        #Finicky, inefficient method to get average norms if not provided
        #Not needed that often so its probably fine
        #ts is kafkaesque
        if not self.vnloaded:
            self.mapping = defaultdict(list)
            for i, face in enumerate(self.faces):
                for v in face.indices:
                    self.mapping[v].append(i)
            self.get_borders()
            print(f"Borders: {time() - t} seconds")
        t = time()

        #Defines center of model
        self.center = self.cc()



    #Face metaclass defines a face (no way)
    class face:
        def __init__(self,outerInstance, indices, id,textureids=None,vn=False):
            #outerInstance is just the model using the face
            #indices are vertex indices
            #textureids and vn (vertex normals) are optional

            self.outerInstance = outerInstance

            textured = False if textureids is None else True
            reverse = outerInstance.reverseNormals

            #id isn't really used
            self.id=id

            #Vertex coordinate
            self.indices = np.array(indices)
            self.points = self.outerInstance.coords[indices]

            if textured:
                #Gets texture points of each vertex if textured
                self.tids = np.array(textureids,dtype=np.int16)
                self.texturepoints = outerInstance.texturecoords[self.tids]

            #Denominator used in converting 3d points to 2d coordinates
            #z+FOV
            denominator = self.points[:, 2] + FOV
            #Converts them
            self.TwoDCoords = np.stack((self.points[:, 0] * FOV / denominator,self.points[:, 1] * FOV / denominator), axis=1)

            #Manually calculates the normal if no vertex normals provided
            if vn == False:
                p0,p1,p2 = outerInstance.coords[indices[:3]]
                v1 = p0-p1
                v2 = p2-p1
                self.normal = np.cross(v1,v2)
            #If vertex normals provided, normal is defined as the average of them
            else:
                self.avNorms = outerInstance.vertexnormals[vn]
                s = sum(self.avNorms)
                self.normal = s/3

            #reverses normal
            if reverse:
                self.normal*=-1

            #normalizes normal
            self.normal /= np.linalg.norm(self.normal)

        #A face is less than another if its z is less than it
        def __lt__(self, other):
            return self.z<other.z


    #Returns center coordinate
    #(just the point in the visual center of the model)
    def cc(self):
        mins = np.min(self.coords, axis=0)
        maxs = np.max(self.coords, axis=0)
        return (mins + maxs) / 2

    #Gets average normals (only used if they are not provided)
    #Absolutely do not remember what i was thinking when making this but it kind of works so thats good enough
    def get_borders(self):
        for face in self.faces:
            face.bordering = [self.mapping[i] for i in face.indices]
            face.avNorms = [np.mean([self.faces[id].normal for id in vertex],axis=0)/np.linalg.norm(np.mean([self.faces[id].normal for id in vertex],axis=0)) for vertex in face.bordering]

    #Recalculates 2d coordinates for all faces
    #Used after transformations
    def update2dCoords(self):
        #Updates them all simultaneously using numpy for speed
        #gets the indices, points, and z from each face
        indices = np.array([face.indices for face in self.faces])
        points = np.array([self.coords[idx] for idx in indices])

        #Denominator for 2d coordinates
        denominator = points[:, :, 2] + FOV

        #Calculates the 2d coords for each point
        TwoDCoords = np.stack((points[:, :, 0] * FOV / denominator, points[:, :, 1] * FOV / denominator), axis=2)

        #Assigns changes to each face again
        for i, face in enumerate(self.faces):
            face.TwoDCoords = TwoDCoords[i]
            face.points = self.coords[face.indices]

        #Puts faces sorted from back to front
        #self.faces.sort(reverse=True)

    #Returns valid faces with regard to backface culling
    def validFaces(self):
        #for some reason, backface culling does not want to work well, so
        return tuple(filter(lambda x: np.dot(x.normal,VIEW_VECTOR)<1e-1, self.faces))

    #Rotates coordinates about a certain axis at its visual center
    def rotateCoords(self,axis: str,  angle:int|float):
        #Axis will be x,y, or z to rotate around the x,y, or z axis
        #x = left to right
        #y = down to up
        #z = back to front

        rotMat = rot_matrix(axis,angle)

        #Shifts coordinates to center, rotates, then unshifts them
        shifted = self.coords - self.center
        shifted = shifted @ rotMat.T
        self.coords = shifted+self.center

        #Rotates the normals
        for idx,face in enumerate(self.faces):
            face.normal = face.normal @ rotMat.T
            face.avNorms = face.avNorms @ rotMat.T

    #Matrix based transformations
    def matrixTransform(self,matrix):
        arr = np.array(matrix)

        #Shifts coordinates to center, rotates, then unshifts them
        shifted = self.coords - self.center
        shifted = shifted @ arr.T
        self.coords = shifted+self.center

        for idx,face in enumerate(self.faces):
            face.normal = face.normal @ arr.T
            face.avNorms = face.avNorms @ arr.T

    #Just shifts the object's coordinates
    def shiftCoords(self, axis: str, amount:int|float):
        # Axis will be x,y, or z to shift in the x,y, or z direction
        # x = left to right
        # y = down to up
        # z = back to front

        axis = axis.lower()
        assert axis in ('x', 'y', 'z'), "Invalid axis, Axis must be 'x','y', or 'z'"
        axis = {'x':0,'y':1,'z':2}[axis]

        shift = np.array([0,0,0])
        shift[axis]=amount

        self.coords += shift
        self.center += shift

    #Scales coordinates about the visual center of the model
    def scaleCoords(self, amount:int|float):
        scale = np.array([amount]*3)

        shifted = self.coords-self.center
        shifted*= scale
        self.coords = shifted + self.center





