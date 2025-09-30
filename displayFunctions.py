import cv2 as cv
import parameters
from shapes import rot_matrix
import numpy as np
from shaders import *
from lightFunctions import *

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