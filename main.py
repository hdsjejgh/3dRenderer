import numpy as np

import parameters
from shapes import *
from shaders import *
import cv2 as cv
#-----------------------------------------------#

#The model loaded
Model = OBJFile("models/Shambler.obj",reverseNormals=True,texture="textures/Shambler.png")

#Transformations to be done to the model before anything
def pretransformation():
    Model.scaleCoords(-3)
    #Model.rotateCoords('x',-90)
    # c.shiftCoords('y', -75)

#What transformations to apply to the model every frame
def TransformationLoop():
    parameters.LIGHT_POS[:] = rot_matrix('y', 2) @ parameters.LIGHT_POS
    parameters.LIGHT_VECTOR[:] = -parameters.LIGHT_POS / np.linalg.norm(parameters.LIGHT_POS)
    #print(parameters.LIGHT_POS)
    #print(parameters.LIGHT_POS)
    #Model.rotateCoords('x', 4)
    #Model.rotateCoords('y', -1)
    #Model.rotateCoords('z', 7)
    #Model.scaleCoords(1.01)



#Display function for nontextured phong
#Separated just for clarity
def display_phong(model):

    #Gets and resets global view vector
    global view

    #Empty zbuffer of infinities
    zbuffer = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float64)

    #Array needed to center points in the display
    #(<0,0> would be shifted to the view's middle)
    centArray = np.array([WIDTH/2,HEIGHT/2])

    #Iterates over faces facing towards camera
    for face in model.validFaces():

        #Centered version of face's 2 dimensional coordinates
        coords = face.TwoDCoords+centArray

        #Rasterizes
        rasterize_phong(
                coords=coords,
                view=view,
                zbuffer=zbuffer,
                av_normals=face.avNorms,
                coords_3d=face.points,
                color=(255,255,255)
        )


#Display function for ntextured phong
#Separated just for clarity
def display_phong_textured(model):
    #Errors if textureless model is given
    assert model.textured, "Shape is not textured"
    texture = model.texture

    # Gets and resets global view vector
    global view

    # Empty zbuffer of infinities
    zbuffer = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float64)

    # Array needed to center points in the display
    # (<0,0> would be shifted to the view's middle)
    centArray = np.array([WIDTH / 2, HEIGHT / 2])

    #Iterates over faces facing towards camera
    for face in model.validFaces():

        #Loads current face's texture coordinates
        texture_points = face.texturepoints

        #Centered version of face's 2 dimensional coordinates
        coords = face.TwoDCoords + centArray

        #Rasterizes with textures
        rasterize_phong_texture(
                coords=coords,
                view=view,
                zbuffer=zbuffer,
                av_normals=face.avNorms,
                coords_3d=face.points,
                texturecoords=texture_points,
                texture=texture,
                LIGHT_POS=parameters.LIGHT_POS,
                LIGHT_VECTOR=parameters.LIGHT_VECTOR
        )



if __name__ == '__main__':

    #T is used to time different things for debugging
    t = time()

    #Performs pretransformations
    pretransformation()

    #How long the pretransformations took
    print(f"Prefunctioning: {time() - t} seconds")
    t = time()

    #Closes display upon clicking 'x' key
    while cv.waitKey(20)&0xff != ord('x'):

        #Updates the 2d coordinates to reflect any transformations
        Model.update2dCoords()

        #View variable is actual display (each element is 1 pixel)
        view = np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8)

        LIGHT2Dx = int((parameters.LIGHT_POS[0] * parameters.FOV) / (parameters.LIGHT_POS[2] + parameters.FOV) + parameters.WIDTH / 2)
        LIGHT2Dy = int((parameters.LIGHT_POS[1] * parameters.FOV) / (parameters.LIGHT_POS[2] + parameters.FOV) + parameters.HEIGHT / 2)


        lights = [(LIGHT2Dx,LIGHT2Dy),(LIGHT2Dx,LIGHT2Dy+1),(LIGHT2Dx,LIGHT2Dy-1),(LIGHT2Dx+1,LIGHT2Dy),(LIGHT2Dx-1,LIGHT2Dy),]
        for x,y in lights:
            if 0<=x<parameters.WIDTH and 0<=y<parameters.HEIGHT:
                view[y, x] = np.array([0, 255, 0]).astype(np.uint8)


        #Updates the view
        #display_phong(Model)
        display_phong_textured(Model)



        #Displays view in opencv
        cv.imshow("3d Render",view)

        #Runs per frame transformations
        TransformationLoop()
