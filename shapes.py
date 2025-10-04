from email.policy import default

from parameters import *
import numpy as np
from time import time
from PIL import Image
from collections import defaultdict
import math
from abc import ABC
#-----------------------------------------------#


#Degrees based sin function
def sin(deg: int|float) -> float:
    return math.sin((deg*math.pi)/180)
#Degrees based cos function
def cos(deg: int|float) -> float:
    return math.cos((deg*math.pi)/180)

#Finds numpy array in a list of numpy arrays
#Needed because numpy is weird with array comparisons
def np_index(arr,target):
    for i in range(len(arr)):
        #Sees if every component is equal
        if (arr[i]==target).all():
            return i

    return -1

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
        rotMat = np.array(
                    [[1, 0, 0],
                           [0, ca, -sa],
                           [0, sa, ca]],
                           dtype=np.float32)
    elif axis == 'y':
        rotMat = np.array(
                    [[ca, 0, sa],
                           [0, 1, 0],
                           [-sa, 0, ca]],
                           dtype=np.float32)
    elif axis == 'z':
        rotMat = np.array(
                    [[ca, sa, 0],
                           [-sa, ca, 0],
                           [0, 0, 1]],
                          dtype=np.float32)

    return rotMat


#The base abstract class for 3d model loading
class File(ABC):

    def __init__(self):
        #All the following must be defined in any file loading classes derived
        #(also just Defining them here just to stop the ide from complaining about this class)
        self.faces = None
        self.coords = None
        self.mapping = None
        self.center = None



        self.valids = []

    #Face metaclass defines a face (no way)
    class face:
        def __init__(self,outerInstance, indices,textureids=None,vn=False,normal = False):
            #outerInstance is just the model using the face
            #indices are vertex indices
            #textureids and vn (vertex normals) are optional

            self.outerInstance = outerInstance

            textured = False if textureids is None else True
            reverse = outerInstance.reverseNormals


            #Vertex coordinate
            self.indices = np.array(indices)
            self.points = self.outerInstance.coords[indices]

            self.center = sum(self.points)/3

            self.z = np.sum(self.points[-1:2])/3

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
            if vn == False and normal is False:
                p0,p1,p2 = outerInstance.coords[indices]
                v1 = p0-p1
                v2 = p2-p1
                self.normal = np.cross(v1,v2)
            #If vertex normals provided, normal is defined as the average of them
            elif normal is False:
                self.avNorms = outerInstance.vertexnormals[vn]
                s = sum(self.avNorms)
                self.normal = s/3
            else:
                self.avNorms = outerInstance.vertexnormals[vn]
                self.normal = np.array(normal,dtype=np.float32)

            #reverses normal
            if reverse:
                self.normal*=-1

            #normalizes normal
            self.normal /= np.linalg.norm(self.normal)

            self.avNorms=self.avNorms.astype(np.float32)

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
    def updateFaces(self):
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
            face.center = sum(face.points)/3

        self.valids = []
        for i,face in enumerate(self.faces):
            # Backface culling
            if CULLING and np.dot(face.normal, VIEW_VECTOR) > 1e-1:
                continue
            # Dont bother rendering a face if its behind the camera
            if np.any(face.points[:, 2] < CAMERA_POS[2]):
                continue
            self.valids.append(i)

    #Returns valid faces with regard to backface culling
    def validFaces(self):
        # for some reason, backface culling does not want to work well for faces near the edge
        # probably because of small rounding errors


        returnFaces = []
        for i in self.valids: returnFaces.append(self.faces[i])

        return returnFaces



    #Linear transformations


    #Rotates coordinates about a certain axis at its visual center
    def rotate(self,axis: str,  angle:int|float, selfcc=True):
        #Axis will be x,y, or z to rotate around the x,y, or z axis
        #x = left to right
        #y = down to up
        #z = back to front

        #selfcc is whether to rotate about its own center coordinates as opposed to the global origin

        rotMat = rot_matrix(axis,angle)
        #Shifts coordinates to center (if selfcc is true), rotates, then unshifts them
        if selfcc:
            shifted = self.coords - self.center
            shifted = shifted @ rotMat.T
            self.coords = shifted+self.center
        else:
            self.coords = self.coords @ rotMat.T

        #Rotates the normals
        for idx,face in enumerate(self.faces):
            face.normal = face.normal @ rotMat.T
            face.avNorms = face.avNorms @ rotMat.T

    #Matrix based transformations
    def matrix_transformation(self,matrix):
        arr = np.array(matrix)

        #Shifts coordinates to center, rotates, then unshifts them
        shifted = self.coords - self.center
        shifted = shifted @ arr.T
        self.coords = shifted+self.center

        arr_inv = np.linalg.inv(arr)

        for idx,face in enumerate(self.faces):
            face.normal = face.normal @ arr_inv
            face.avNorms = face.avNorms @ arr_inv

    #Just shifts the object's coordinates
    def shift(self, axis: str, amount:int|float):
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

    #Shifts model so it is centered at the origin
    def centerShift(self):
        shift = self.center
        self.coords -= shift
        self.center -= shift


    #Scales coordinates about the visual center of the model
    def scale(self, amount:int|float,cent=None):
        if cent is None: cent = self.center
        else: cent = np.array(cent)

        scale = np.array([amount]*3)

        shifted = self.coords-cent
        shifted*= scale
        self.coords = shifted + cent


    #Nonlinear transformations


    #Performs a nonlinear twist transformation
    def twist(self,axis,deg,constant,center=0):
        #axis is axis of rotation
        #deg is the base degree of rotation
        #constant is the twist constant in the degree_final = deg + constant * vector[axis] equation
        #Farther from the origin (or center) a point is, the more it gets twisted

        #nested function :wilted_rose:
        #ts is orwellian

        axis = axis.lower()
        #Which component corresponds with the axis
        index = {'x':0,'y':1,'z':2}[axis]

        def twistMat(vect):
            #Input a vector, output an appropriately twisted one

            component = vect[index]
            #the actual angle of rotation based on the twist formula
            angle = deg+constant*(component-center)

            rot = rot_matrix(axis,angle)
            return rot @ vect

        def twist_normal(vect,component):
            # Input a normal vector, output an appropriately twisted one

            # the actual angle of rotation based on the twist formula
            angle = deg + constant * (component - center)
            rot = rot_matrix(axis, angle)
            return rot @ vect

        #vectorized version of twist function
        #inputs a vector, outputs a twisted one
        vectorized = np.vectorize(twistMat,signature='(n)->(n)')

        #Twists coordinates
        self.coords = vectorized(self.coords)

        #Twists average normals and normals for each face
        for idx,face in enumerate(self.faces):
            for i in range(3):
                comp = face.points[i,index]
                face.avNorms[i] = twist_normal(face.avNorms[i],comp)

            component = face.center[index]
            face.normal = twist_normal(face.normal,component)


    # Performs a nonlinear taper transformation
    def linear_taper(self, axis, a_constant,a_coeff,b_constant,b_coeff):
        #(given axis is ignored in transformation)
        #Transformation matrix is diagonal with a_constant + p_1 * a_coeff for the first non-axis term
        #and that but b for the second

        axis = axis.lower()
        if axis == 'x':
            xEquation = "1"
            yEquation = "a_constant + a_coeff*c"
            zEquation = "b_constant + b_coeff*c"
        elif axis == 'y':
            yEquation = "1"
            xEquation = "a_constant + a_coeff*c"
            zEquation = "b_constant + b_coeff*c"
        elif axis == 'z':
            zEquation = "1"
            xEquation = "a_constant + a_coeff*c"
            yEquation = "b_constant + b_coeff*c"
        else:
            raise Exception(f"Invalid axis: {axis}")

        # Which component corresponds with the axis
        index = {'x': 0, 'y': 1, 'z': 2}[axis]

        def taper(vect):
            # Input a point, output an appropriately tapered one

            c = vect[index]

            # The eval doesn't seem to work unless the parameters are mentioned, this seems to fix it
            a_constant; a_coeff; b_constant; b_coeff

            mat = np.array([
                [eval(xEquation),0,0],
                [0,eval(yEquation),0],
                [0,0,eval(zEquation)],
            ])

            return mat @ vect

        def taper_normal(vect,component):
            # Input a normal and corresponding component, output an appropriately tapered one

            c = vect[index]

            #The eval doesn't seem to work unless the parameters are mentioned????, this seems to fix it
            a_constant; a_coeff; b_constant; b_coeff

            mat = np.array([
                [eval(xEquation), 0, 0],
                [0, eval(yEquation), 0],
                [0, 0, eval(zEquation)],
            ])

            #Uses inverse transpose to transform normal
            mat = np.linalg.inv(mat).T

            return mat @ vect

        # vectorized version of twist function
        # inputs a vector, outputs a twisted one
        vectorized = np.vectorize(taper, signature='(n)->(n)')

        # Twists coordinates
        self.coords = vectorized(self.coords)

        for idx,face in enumerate(self.faces):
            for i in range(3):
                comp = face.points[i,index]
                face.avNorms[i] = taper_normal(face.avNorms[i],comp)

            component = face.center[index]
            face.normal = taper_normal(face.normal,component)


#File Loaders


#Class for loading .OBJ Files
class OBJ_File(File):

    def __init__(self,filepath, reverseNormals=False,texture=None, *args, **kwargs):
        # Confirms file type
        extension = filepath.strip().split('.')[-1].lower()
        assert extension == "obj", f"The file path {filepath} is not an obj file"

        super().__init__()
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
                self.faces = [self.face(self,
                                        self.faces[i],
                                        textureids = self.textureids[i])
                              for i in range(len(self.faces))]
            else:
                self.faces = [self.face(self,
                                        self.faces[i])
                              for i in range(len(self.faces))]

        #If vertex normals are provided, faces won't have to calculate them (quicker and more accurate)
        else:
            if self.textured:
                self.faces = [self.face(self,
                                        self.faces[i],
                                        textureids = self.textureids[i],
                                        vn=self.vnids[i])
                              for i in range(len(self.faces))]
            else:
                self.faces = [self.face(self,
                                        self.faces[i],
                                        vn=self.vnids[i])
                              for i in range(len(self.faces))]

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


class STL_ASCII_File(File):

    def __init__(self,filepath, reverseNormals=False, texture=None, *args, **kwargs):
        #Confirms file type
        extension = filepath.strip().split('.')[-1].lower()
        assert extension == "stl", f"The file path {filepath} is not an stl file"

        super().__init__()
        # Holds 3d coordinates of file
        self.coords = []

        self.reverseNormals = reverseNormals

        # Holds the variety of face objects
        self.faces = []

        with open(filepath,'r') as file:
            header = next(file)
            #All ascii based .stl files start with "solid"
            assert header.split()[0]=="solid", "File is not an ASCII based .STL file"

            #index of faces, variable needed since faces don't get stored at first
            f = 0

            #Face normals (corresponds with index variable f above)
            normals = []

            #Maps each vertex with which faces it is connected to
            vert2face = defaultdict(list)

            #First iteration through file is to generate all the vertex normals since stl files do not include them
            for i,line in enumerate(file):
                line = line.strip().split()

                match line[0]:
                    #If line describes a face
                    case "facet":
                        #Adds normal to normals list in such a way that the ith entry is for the ith face
                        normals.append(np.array(line[-3:]))
                    #If line describes a vertex
                    case "vertex":
                        #The coordinates of the vertex
                        v = line[-3:]
                        vert2face[tuple(v)].append(f)
                        self.coords.append(v)
                    #If line describes the end of a face
                    case "endfacet":
                        f+=1

            normals = np.array(normals,dtype=np.float32)

            #Maps each vertex to its normal
            vert2norm = {v:sum(normals[vert2face[v]])/len(vert2face[v]) for v in vert2face}
            #List of vertex normals
            self.vertexnormals = np.array(list(vert2norm.values()),)

            #Restarts file
            file.seek(0)

            #Vertex indices per face
            vindices = []
            #Vertex normal (avg normal) indices per face
            avgnindices = []
            #Normal per face
            normal = []

            c = self.coords
            self.coords = np.array(self.coords,dtype = np.float32)

            #Second iteration through file actually creates faces
            #Incredibly inefficient :wilted_rose:
            for i,line in enumerate(file):
                line = line.strip().split()

                match line[0]:
                    # If line describes a face
                    case "facet":
                        #Gets face normal
                        normal = np.array(line[-3:])
                    #If a line describes a vertex
                    case "vertex":
                        v = np.array(line[-3:])
                        #Gets the index of the vertex in the list of vertices
                        vid = np_index(c,v)
                        vindices.append(vid)

                        #the vertex normal of the current vertex
                        vn = np.array(vert2norm[tuple(v)])
                        #The index of the vn in the list of vn's is recorded
                        avgnindices.append(np_index(self.vertexnormals,vn))
                    #If a line describes the end of a face
                    case "endfacet":
                        #Creates face based on recorded information
                        self.faces.append(self.face(self,
                                                    indices=vindices,
                                                    vn=avgnindices,
                                                    normal = normal)
                                          )

                        #Resets everything to record for the next face
                        vindices = []
                        avgnindices = []
                        normal = []

        #Finds model's visual center
        self.center = self.cc()



