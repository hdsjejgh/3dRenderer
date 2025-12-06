from shapes import *
from displayFunctions import *
from lightFunctions import *
#-----------------------------------------------#

cv.namedWindow("3d Render")

#Previous mouse coordinates; used to calculate mouse movement
prevx,prevy = -1,-1


#Font info for the display
font = cv.FONT_HERSHEY_SIMPLEX
#Bottom left of the first display thing (The fps)
org = np.array([10,25])
#Font visual options
fontScale = .5
color = (0, 255, 0)
thickness = 2
lineType = cv.INTER_AREA

#Average FPS, alpha, and current timestep for calculating the fps using exponentially moving average
averageFPS = 0.0
ALPHA = 0.9
timestep = 0

#Number of valid faces in current frame (for diagnostic purposes)
num_faces = 0

#Handles all mouse functions
#Dragging rotates everything in the expected manner
#Scrolling up zooms in
#Scrolling down zooms out
def mouse_callback(event, x, y, flags, params):
    global Model
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

            for i in range(len(LIGHTS_INFO)):
                lightRot('y',(-dx/parameters.WIDTH)*180,i,LIGHTS_INFO)
            Model.rotate('y',(-dx/parameters.WIDTH)*180,selfcc=False)

            for i in range(len(LIGHTS_INFO)):
                lightRot('x', (dy / parameters.HEIGHT) * 180,i,LIGHTS_INFO)
            Model.rotate('x', (dy / parameters.HEIGHT) * 180,selfcc=False)

            Model.updateFaces()

    #If mouse scrolled
    elif event == cv.EVENT_MOUSEWHEEL:
        #If scrolled up
        if flags>0:
            Model.scale(1.1,[0,0,0])
            for i in range(len(LIGHTS_INFO)):
                lightScale(1.1,i,LIGHTS_INFO)
        #If scrolled down
        elif flags<0:
            Model.scale(1/1.1,[0,0,0])
            for i in range(len(LIGHTS_INFO)):
                lightScale(1/1.1,i,LIGHTS_INFO)

#Sets up mouse callback function
cv.setMouseCallback("3d Render",mouse_callback)


#CHANGE THESE THINGS ##########################################

#Pass this as the light info array in any light creation, editing function
LIGHTS_INFO =  [[],[]]

#The model loaded
Model = OBJ_File("models/Shambler.obj",reverseNormals=True,)#texture="textures/Shambler.png")
# Which shader rasterizing function to use
# All are in the displayFunctions file
SHADER = phong

#Transformations to be done to the model before anything
def pretransformation():
    pass
    createLight(200,0,0,LIGHTS_INFO,0.25)
    createLight(-200, 0, 0, LIGHTS_INFO)

    Model.scale(-3)
    #Model.rotate('x',-90)
    Model.centerShift()

#What transformations to apply to the model every frame
def TransformationLoop():
    pass

    # Model.linear_taper('y',1.00001,0.0001,1.00001,0.0001)
    # Model.twist('x', 1, 0.01, center=100)
    #Model.twist('y', 1, 0.01, center=100)
    lightRot('y',1,0,LIGHTS_INFO)
    lightRot('y', 1, 1, LIGHTS_INFO)


#######################################################


if __name__ == '__main__':

    #T is used to time different things for debugging
    t = time()

    #Performs pretransformations
    pretransformation()

    #How long the pretransformations took
    print(f"Prefunctioning: {time() - t} seconds")
    t = time()

    #Closes display upon clicking 'x' key
    while cv.waitKey(1)&0xff != ord('x'):

        timestep += 1

        #Updates the 2d coordinates to reflect any transformations
        Model.updateFaces()

        #View variable is actual display (each element is 1 pixel)
        view = np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8)

        zbuffer = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)

        # Updates the view
        SHADER(Model, view, zbuffer,LIGHTS_INFO)

        #Adds a white dot to the display and zbuffer indicate the origin
        origin = (int(parameters.WIDTH/2),int(parameters.HEIGHT/2))
        if zbuffer[origin[1],origin[0]] >=0:
            view[origin[1],origin[0]] = np.array([255, 255, 255]).astype(np.uint8)
            zbuffer[origin[1], origin[0]] = 0

        intensities = LIGHTS_INFO[1]
        for i,LIGHT_POS in enumerate(LIGHTS_INFO[0]):
            #Skips light if its behind model
            if LIGHT_POS[2]>parameters.CAMERA_POS[2]:

                #Adds a yellow square (5x5) to the view and zbuffer to indicate the light source
                LIGHT2Dx = int((LIGHT_POS[0] * parameters.FOV) / (LIGHT_POS[2] + parameters.FOV) + parameters.WIDTH / 2)
                LIGHT2Dy = int((LIGHT_POS[1] * parameters.FOV) / (LIGHT_POS[2] + parameters.FOV) + parameters.HEIGHT / 2)

                lightradius = 10
                lights = [(LIGHT2Dx+i,LIGHT2Dy+j) for i in range(-lightradius+1,lightradius) for j in range(-lightradius+1,lightradius)]
                for x,y in lights:
                    if 0<=x<parameters.WIDTH and 0<=y<parameters.HEIGHT and zbuffer[y,x]>LIGHT_POS[2]:
                        #Makes pixels further from the center of the light darker since it looks better
                        #Does make said pixels black even when above the model, but its a minor thing, ill fix it later
                        dist = (x-LIGHT2Dx)**2+(y-LIGHT2Dy)**2
                        light_color = intensities[i]*255*.75**(dist/3)

                        view[y, x] = np.clip(view[y, x]+np.array([light_color, light_color, light_color]),a_min=0,a_max=255).astype(np.uint8)
                        zbuffer[y,x] = LIGHT_POS[2]

        if parameters.AA=="FXAA":
            view = FXAA(view,parameters.LUM_VECT)


        #Calculating and displaying new exponentially weighted average
        FPS = 1/(time() - t)
        averageFPS = ALPHA * averageFPS + (1-ALPHA) * FPS
        #Applies bias correction so that initial values aren't low
        #The *5 isn't part of the formula it just happens to make the calculations more stable so im including it
        averageFPS /= 1-(ALPHA**(timestep*5))
        cv.putText(view,f"{averageFPS:.1f} FPS",org, font, fontScale, color, thickness, lineType)

        #Displays current number of faces
        num_faces = len(Model.valids)
        cv.putText(view, f"{num_faces} Faces", org+np.array([0,20]), font, fontScale, color, thickness, lineType)

        #Displays view in opencv
        cv.imshow("3d Render",view)

        t = time()

        #Runs per frame transformations
        TransformationLoop()
