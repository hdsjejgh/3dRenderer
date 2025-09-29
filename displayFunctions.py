import cv2 as cv
import parameters
from shapes import rot_matrix
import numpy as np
from shaders import *
from lightFunctions import *


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

            lightRot('y',(-dx/parameters.WIDTH)*180)
            Model.rotate('y',(-dx/parameters.WIDTH)*180,selfcc=False)

            lightRot('x', (dy / parameters.HEIGHT) * 180)
            Model.rotate('x', (dy / parameters.HEIGHT) * 180,selfcc=False)

            Model.updateFaces()

    #If mouse scrolled
    elif event == cv.EVENT_MOUSEWHEEL:
        #If scrolled up
        if flags>0:
            Model.scale(1.1,[0,0,0])
            lightScale(1.1)
        #If scrolled down
        elif flags<0:
            Model.scale(1/1.1,[0,0,0])
            lightScale(1/1.1)


#Shader rasterization functions


#Display function for phong shading
def phong(model,view,zbuffer):

    global num_faces

    #Array needed to center points in the display
    #(<0,0> would be shifted to the view's middle)
    centArray = np.array([parameters.WIDTH/2,parameters.HEIGHT/2])

    validFaces = model.validFaces()

    #Saves number of faces in current frame
    num_faces = len(validFaces)

    #If model is untextured
    if not model.textured:
        #Iterates over faces facing towards camera to render
        for face in validFaces:

            #Centered version of face's 2 dimensional coordinates
            coords = face.TwoDCoords+centArray

            #Rasterizes
            rasterize_phong(
                    coords=coords,
                    view=view,
                    zbuffer=zbuffer,
                    av_normals=face.avNorms,
                    coords_3d=face.points,
                    color=(255,255,255),
                    LIGHT_POS=parameters.LIGHT_POS,
                    LIGHT_VECTOR=parameters.LIGHT_VECTOR
            )

    #If model is textured
    if model.textured:
        texture = model.texture
        for face in model.validFaces():
            # Loads current face's texture coordinates
            texture_points = face.texturepoints

            # Centered version of face's 2 dimensional coordinates
            coords = face.TwoDCoords + centArray

            # Rasterizes with textures
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