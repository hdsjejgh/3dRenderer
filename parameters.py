import numpy as np
#-----------------------------------------------#

#Display Parameters

WIDTH = 800
HEIGHT = 800
FOV = 250
#Whether or not to use ZBuffer (A bit finicky)
ZBUFF = True
#Whether or not to use Backface Culling
CULLING = True

#Anti-Aliasing parameters

#Luminance vector (dotted with color to find luminance)
LUM_VECT = np.array([(0.587/0.299),1,0]).astype(np.float32)/255
#Minimum local contrast to apply FXAA
FXAA_EDGE_THRESHOLD = 1/8
FXAA_EDGE_THRESHOLD_MIN = 1/16

FXAA_SUBPIX = 1.0
FXAA_SUBPIX_TRIM = 1/4
FXAA_SUBPIX_CAP = 3/4
FXAA_SUBPIX_TRIM_SCALE = 1/(1-FXAA_SUBPIX_TRIM)

FXAA_SEARCH_STEPS = 5
FXAA_SEARCH_ACCELERATION = 1
FXAA_SEARCH_THRESHOLD = 1/4

#Global Vectors

#The vector representing camera direction
VIEW_VECTOR = np.array([0,0,1],dtype = 'float32')
#The camera position vector
#Technically not what it actually is in the display (its closer to <0,0,-200>), but it being closer than it makes the phong lighting look nicer
CAMERA_POS = np.array([0,0,-200],dtype = 'float32')
#The light source position vector
LIGHT_POS = np.array([200,0,0],dtype = 'float32')
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
AMBIENT_INTENSITY = 0.05*255
#Gamma value for gamma correction
#Higher = darker base color
GAMMA = 1


