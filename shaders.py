import random
import math
from parameters import BACKFACECULLING

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

def sideShadow():
    assert BACKFACECULLING, "Backface Culling must be on for Side Shadow"
    def wrapper(face):
        return tuple(205*face.normal[2]**2+50 for i in range(3))
    return wrapper