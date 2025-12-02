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

#What kind of antialiasing to run
AA = {
    0:None,
    1:"FXAA"
#change the key to change the option
}[0]

#Luminance vector (dotted with color to find luminance)
LUM_VECT = np.array([0.299, 0.587, 0.114]).astype(np.float32)
#Minimum local contrast to apply FXAA
FXAA_EDGE_THRESHOLD = 1/4
#1/3 - too little
#1/4 - low quality
#1/8 - high quality
#1/16- overkill

#stops processing of darks
FXAA_EDGE_THRESHOLD_MIN = 1/32
#1/32 - visible limit
#1/16 - high quality
#1/12 - upper limit

#FXAA_SUBPIX = 1.0
#controls removal of subpix aliasing
FXAA_SUBPIX_TRIM = 1/4
#1/2 - low removal
#1/3 - medium removal
#1/4 - default removal
#1/8 - high removal
#0   - complete removal

#partly overrides the above
FXAA_SUBPIX_CAP = 3/4
#3/4 - default amount of filtering
#7/8 - high amount of filtering
#1   - no capping or filtering

FXAA_SUBPIX_TRIM_SCALE = 1/(1-FXAA_SUBPIX_TRIM)

#max search steps
FXAA_SEARCH_STEPS = 3
#step size in search
FXAA_SEARCH_ACCELERATION = 1
#controls when to stop searching
FXAA_SEARCH_THRESHOLD = 1/4

#Global Vectors

#The vector representing camera direction
VIEW_VECTOR = np.array([0,0,1],dtype = 'float32')
#The camera position vector
#Technically not what it actually is in the display (its closer to <0,0,-200>), but it being closer than it makes the phong lighting look nicer
CAMERA_POS = np.array([0,0,-200],dtype = 'float32')
#The light source position vector
LIGHT_POS = np.array([200,0,0],dtype = 'float32')




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


