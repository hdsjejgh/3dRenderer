import numpy as np
#-----------------------------------------------#

#Display Parameters

WIDTH = 800
HEIGHT = 800
FPS = 30
FOV = 250
#Whether or not to use ZBuffer (A bit finicky)
ZBUFF = True
#Whether or not to use Backface Culling
CULLING = False


#Global Vectors

#The vector representing camera direction
VIEW_VECTOR = np.array([0,0,1],dtype = 'float64')
#The camera position vector
#Technically not what it actually is in the display (its closer to <0,0,-200>), but it being closer than it makes the phong lighting look nicer
CAMERA_POS = np.array([0,0,-200],dtype = 'float64')
#The light source position vector
LIGHT_POS = np.array([200,0,0],dtype = 'float64')
#The light direction vector (set to face the origin)
LIGHT_VECTOR = -LIGHT_POS/np.linalg.norm(LIGHT_POS)


#Lighting Parameters

#Phong exponent for phong lighting
#Higher = Sharper and smaller shine
#Lower = Dispersed shine
PHONG_EXPONENT = 200
#Controls overall shine
REFLECTIVITY_CONSTANT = 3
#Ambient lighting for phong
AMBIENT_INTENSITY = 0.25*255
#Gamma value for gamma correction
#Higher = darker base color
GAMMA = 1


