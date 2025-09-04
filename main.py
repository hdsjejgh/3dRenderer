import parameters
from shapes import *
from shaders import *
import cv2 as cv
#-----------------------------------------------#

cv.namedWindow("3d Render")

#Previous mouse coordinates; used to calculate mouse movement
prevx,prevy = -1,-1

#Handles all mouse functions
#Dragging rotates everything in the expected manner
#Scrolling up zooms in
#Scrolling down zooms out
def mouse_callback(event, x, y, flags, params):
    global prevx,prevy

    #If mouse pressed
    if event == cv.EVENT_LBUTTONDOWN:
        prevx,prevy = x,y
    #If mouse unpressed, erase previous position data
    elif event == cv.EVENT_LBUTTONUP:
        prevx,prevy = -1,-1
    #If mouse moved and previous position data exists, rotate everything
    elif event == cv.EVENT_MOUSEMOVE:
        if prevx!=-1 and prevy !=-1:
            dx = x-prevx
            dy = y-prevy
            prevx = x
            prevy = y

            #Rotations are scaled such that dragging across half the screen rotats the model 90 degrees

            lightRot('y',(-dx/parameters.WIDTH)*180)
            Model.rotateCoords('y',(-dx/parameters.WIDTH)*180,selfcc=False)

            lightRot('x', (dy / parameters.HEIGHT) * 180)
            Model.rotateCoords('x', (dy / parameters.HEIGHT) * 180,selfcc=False)

            Model.update2dCoords()

    #If mouse scrolled
    elif event == cv.EVENT_MOUSEWHEEL:
        #If scrolled up
        if flags>0:
            Model.scaleCoords(1.1)
            lightScale(1.1)
        #If scrolled down
        elif flags<0:
            Model.scaleCoords(1/1.1)
            lightScale(1/1.1)

#Sets up mouse callback function
cv.setMouseCallback("3d Render",mouse_callback)

#The model loaded
Model = OBJFile("models/ZombieFilth.obj",reverseNormals=True,texture="textures/ZombieFilth.png")

#Rotations the light source about a given access (x,y,z) by a given number of degrees
def lightRot(axis,deg):
    parameters.LIGHT_POS[:] = rot_matrix(axis, deg) @ parameters.LIGHT_POS
    parameters.LIGHT_VECTOR[:] = -parameters.LIGHT_POS / np.linalg.norm(parameters.LIGHT_POS)

#Scales the light position vector by given magnitude
def lightScale(magnitude):
    parameters.LIGHT_POS*=magnitude
    if magnitude<0:
        parameters.LIGHT_VECTOR *= -1

#Transformations to be done to the model before anything
def pretransformation():
    Model.scaleCoords(-1)
    #Model.rotateCoords('x',-90)
    Model.shiftCoords('y', -75)

#What transformations to apply to the model every frame
def TransformationLoop():
    pass
    lightRot('y',2)
    #print(parameters.LIGHT_POS)
    #print(parameters.LIGHT_POS)
    #Model.rotateCoords('x', 4)
    #Model.rotateCoords('y', 2)
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
    global zbuffer

    # Empty zbuffer of infinities


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

        zbuffer = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float64)

        #Adds a white dot to the display and zbuffer indicate the origin
        origin = (int(parameters.WIDTH/2),int(parameters.HEIGHT/2))
        view[origin[1],origin[0]] = np.array([255, 255, 255]).astype(np.uint8)
        zbuffer[origin[1], origin[0]] = 0

        #Adds a yellow square (5x5) to the view and zbuffer to indicate the light source
        LIGHT2Dx = int((parameters.LIGHT_POS[0] * parameters.FOV) / (parameters.LIGHT_POS[2] + parameters.FOV) + parameters.WIDTH / 2)
        LIGHT2Dy = int((parameters.LIGHT_POS[1] * parameters.FOV) / (parameters.LIGHT_POS[2] + parameters.FOV) + parameters.HEIGHT / 2)

        lights = [(LIGHT2Dx+i,LIGHT2Dy+j) for i in range(-2,3) for j in range(-2,3)]
        for x,y in lights:
            if 0<=x<parameters.WIDTH and 0<=y<parameters.HEIGHT:
                view[y, x] = np.array([0, 255, 255]).astype(np.uint8)
                zbuffer[y,x] = parameters.LIGHT_POS[2]


        #Updates the view
        #display_phong(Model)
        display_phong_textured(Model)



        #Displays view in opencv
        cv.imshow("3d Render",view)

        #Runs per frame transformations
        TransformationLoop()
