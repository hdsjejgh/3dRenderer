import cv2 as cv
import parameters
from shapes import rot_matrix
import numpy as np
from shaders import *
from lightFunctions import *

#Shader rasterization functions


#Display function for lambertian shading
def lambertian(model,view,zbuffer,light_info):
    lights, intensities,colors = light_info

    #Array needed to center points in the display
    #(<0,0> would be shifted to the view's middle)
    centArray = np.array([parameters.WIDTH/2,parameters.HEIGHT/2])

    validFaces = model.validFaces()

    if not model.textured:

        #Iterates over faces facing towards camera to render
        for face in validFaces:

            #Centered version of face's 2 dimensional coordinates
            coords = face.TwoDCoords+centArray


            #Rasterizes
            rasterize_lambertian_textureless(
                coords=coords,
                view=view,
                zbuffer=zbuffer,
                coords_3d=face.points,
                normal=face.normal,
                LIGHTS = lights,
                intensities=intensities,
                colors=colors

            )
    elif model.textured:
        texture = model.texture
        for face in model.validFaces():
            # Loads current face's texture coordinates
            texture_points = face.texturepoints

            # Centered version of face's 2 dimensional coordinates
            coords = face.TwoDCoords + centArray

            # Rasterizes with textures
            rasterize_lambertian_textured(
                coords=coords,
                view=view,
                zbuffer=zbuffer,
                normal=face.normal,
                coords_3d=face.points,
                texturecoords=texture_points,
                texture=texture,
                LIGHTS = np.array(lights),
                intensities=intensities,
                colors=colors
            )


#Display function for gouraud shading
def gouraud(model,view,zbuffer,light_info):
    lights, intensities,colors = light_info

    #Array needed to center points in the display
    #(<0,0> would be shifted to the view's middle)
    centArray = np.array([parameters.WIDTH/2,parameters.HEIGHT/2])

    validFaces = model.validFaces()

    if not model.textured:

        #Iterates over faces facing towards camera to render
        for face in validFaces:

            #Centered version of face's 2 dimensional coordinates
            coords = face.TwoDCoords+centArray

            #Rasterizes
            rasterize_gouraud_textureless(
                coords=coords,
                view=view,
                zbuffer=zbuffer,
                coords_3d=face.points,
                normals=face.avNorms,
                LIGHTS=lights,
                intensities=intensities,
                colors=colors

            )

    elif model.textured:
        texture = model.texture
        for face in model.validFaces():
            # Loads current face's texture coordinates
            texture_points = face.texturepoints

            # Centered version of face's 2 dimensional coordinates
            coords = face.TwoDCoords + centArray

            # Rasterizes with textures
            rasterize_gouraud_textured(
                coords=coords,
                view=view,
                zbuffer=zbuffer,
                normals=face.avNorms,
                coords_3d=face.points,
                texturecoords=texture_points,
                texture=texture,
                LIGHTS=lights,
                intensities=intensities,
                colors=colors
            )





#Display function for phong shading
def phong(model,view,zbuffer,light_info):
    lights, intensities,colors = light_info

    #Array needed to center points in the display
    #(<0,0> would be shifted to the view's middle)
    centArray = np.array([parameters.WIDTH/2,parameters.HEIGHT/2])

    validFaces = model.validFaces()

    #If model is untextured
    if not model.textured:
        #Iterates over faces facing towards camera to render
        for face in validFaces:

            #Centered version of face's 2 dimensional coordinates
            coords = face.TwoDCoords+centArray

            #Rasterizes
            rasterize_phong_textureless(
                    coords=coords,
                    view=view,
                    zbuffer=zbuffer,
                    av_normals=face.avNorms,
                    coords_3d=face.points,
                    LIGHTS=lights,
                    intensities=intensities,
                    colors=colors
            )

    #If model is textured
    elif model.textured:
        texture = model.texture
        for face in model.validFaces():
            # Loads current face's texture coordinates
            texture_points = face.texturepoints

            # Centered version of face's 2 dimensional coordinates
            coords = face.TwoDCoords + centArray

            # Rasterizes with textures
            rasterize_phong_textured(
                coords=coords,
                view=view,
                zbuffer=zbuffer,
                av_normals=face.avNorms,
                coords_3d=face.points,
                texturecoords=texture_points,
                texture=texture,
                LIGHTS=lights,
                intensities = intensities,
                colors = colors

            )