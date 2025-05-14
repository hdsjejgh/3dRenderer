import random
import math
from parameters import *
import numpy as np

#(for the random shader)
COLORS = tuple(tuple(random.randint(0,255) for ii in range(3)) for i in range(100))

#Fragment Shaders

def randomShader(): #assigns random color to each face (based on id), face stays that random color
    def wrapper(face):
        return COLORS[face.id%len(COLORS)]
    return wrapper

#Shader that mades faces darker furter away, and lighter closer
def distShader(mult=0.05,base=(255,255,255)): #mult controls steepness of sigmoid function
    #steeper mult = steeper sigmoid = quicker dropoff of brightness
    def wrapper(face):
        val = 255 * (1-  (1/(1+math.exp(-mult*face.z)))   )
        val = min(255,val)
        #print(val)
        return tuple(map(lambda x:x*(val/255),base))
    return wrapper

def Lambertian(): #technically this isnt lambertian but its close enough
    assert BACKFACECULLING, "Backface Culling must be on for Lambertian"
    def wrapper(face):
        return np.array([205*(np.dot(VIEW_VECTOR,face.normal))**2+50]*3, dtype=np.uint8)
    return wrapper

def Gouraud():
    def wrapper(point,params,points3d,normals):
        A,B,C,D = params
        x,y = point
        z = (D - A*x - B*y) / C
        points = np.array([x,y,z])
        dists = np.linalg.norm(points3d - point, axis=1)
        colors = [np.array([205*(np.dot(VIEW_VECTOR,normal))**2+50]*3, dtype=np.uint8) for normal in normals]
        color = sum(colors[i]/dists[i] for i in range(3))*np.mean(dists)
        return color
    return wrapper


