from shapes import *
from parameters import *
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
    #Model.rotateCoords('x', 4)
    Model.rotateCoords('y', -3)
    #Model.rotateCoords('z', 7)

#Display function for nontextured phong
#Separated just for clarity
def display_phong(model):

    #Gets and resets global view vector
    global view
    view = np.zeros((HEIGHT, WIDTH, 3),dtype=np.uint8)

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
    view = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

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
                texture=texture
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

        #Updates the view
        #display_phong(Model)
        display_phong_textured(Model)

        #Displays view in opencv
        cv.imshow("3d Render",view)

        #Runs per frame transformations
        TransformationLoop()
