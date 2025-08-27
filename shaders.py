import random
import math
from parameters import *
import numpy as np
import numba


#(for the random shader)
COLORS = tuple(tuple(random.randint(0,255) for ii in range(3)) for i in range(100))

#Fragment Shaders

def randomShader(): #assigns random color to each face (based on id), face stays that random color
    def wrapper(face):
        return COLORS[face.id%len(COLORS)]
    return wrapper

#Shader that mades faces darker furter away, and lighter closer
def distShader(mult=0.05,base=(255,255,255)): #mult controls steepness of sigmoid function
    #steeper mult = steeper sigmoid = quicker dropoff of brightness
    def wrapper(face):
        val = 255 * (1-  (1/(1+math.exp(-mult*face.z)))   )
        val = min(255,val)
        #print(val)
        return tuple(map(lambda x:x*(val/255),base))
    return wrapper

def Lambertian(): #technically this isnt lambertian but its close enough
    def wrapper(face):
        return np.array([205*(np.dot(VIEW_VECTOR,face.normal))**2+50]*3, dtype=np.uint8)
    return wrapper

def Gouraud():
    def wrapper(point,params,points3d,normals):
        A,B,C,D = params
        x,y = point
        z = (D - A*x - B*y) / C
        points = np.array([x,y,z])
        dists = np.linalg.norm(points3d - point, axis=1)
        colors = [np.array([205*(np.dot(VIEW_VECTOR,normal))**2+50]*3, dtype=np.uint8) for normal in normals]
        color = sum(colors[i]/dists[i] for i in range(3))*np.mean(dists)
        return color
    return wrapper


@numba.njit()
def rasterize_phong_texture(coords, view, zbuffer, av_normals, coords_3d, normal,color = (255,255,255), texturecoords=None, texture=np.empty((1,1),np.int64)):
    base = np.array(color, dtype=np.float64)
    red,green,blue = base
    A, B, C = coords.astype(np.float64)
    n1, n2, n3 = [np.asarray(n, dtype=np.float64) for n in av_normals]
    coords_3d = coords_3d.astype(np.float64)
    base = np.array(color,dtype=np.float64)
    #The 3d coordinated
    x1, y1, z1 = coords_3d[0]
    x2, y2, z2 = coords_3d[1]
    x3, y3, z3 = coords_3d[2]

    #Used for the bounding box
    min_x = max(int(min(A[0], B[0], C[0])), 0)
    max_x = min(int(max(A[0], B[0], C[0])) + 1, view.shape[1])
    min_y = max(int(min(A[1], B[1], C[1])), 0)
    max_y = min(int(max(A[1], B[1], C[1])) + 1, view.shape[0])

    #area of the triangle face
    v1, v2 = A - B, C - B
    total_area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

    if total_area == 0: #if triangle is too small, just skip the face
        return

    #denominator used to calculate for barycentric weight incrementation
    denom = (B[0] - C[0]) * (A[1] - C[1]) - (B[1] - C[1]) * (A[0] - C[0])

    #alpha and beta for the first (top left) point in bounding box
    x0,y0=min_x,min_y
    alpha_0 = ((B[0] - C[0]) * (y0 - C[1]) - (B[1] - C[1]) * (x0 - C[0])) / denom
    beta_0 = ((C[0] - A[0]) * (y0 - A[1]) - (C[1] - A[1]) * (x0 - A[0])) / denom

    #change in barycentric weights per change in x/y
    dalpha_x = -(B[1] - C[1]) / denom
    dalpha_y = (B[0] - C[0]) / denom
    dbeta_x = -(C[1] - A[1]) / denom
    dbeta_y = (C[0] - A[0]) / denom

    #diffuse lighting from all 3 average vertices calculated ahead of time
    d1 = max(n1.dot(LIGHT_VECTOR), 0)
    d2 = max(n2.dot(LIGHT_VECTOR), 0)
    d3 = max(n3.dot(LIGHT_VECTOR), 0)

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):

            #calculates 2d barycentric weights using previously found change values
            alpha = alpha_0 + (x - min_x) * dalpha_x + (y - min_y) * dalpha_y
            beta = beta_0 + (x - min_x) * dbeta_x + (y - min_y) * dbeta_y
            gamma = 1 - alpha - beta

            if texture.size != 0:
                tc = alpha * texturecoords[0] + beta * texturecoords[1] + gamma * texturecoords[2]

                i = int(round(tc[1])) - 1
                j = int(round(tc[0])) - 1
                i = max(0, min(texture.shape[0] - 1, i))
                j = max(0, min(texture.shape[1] - 1, j))

                base = np.asarray(texture[i, j], dtype=np.float64)


            if alpha>=0 and beta>=0 and gamma>=0: #if the current point is in the triangle, continue
                surface_point = alpha * coords_3d[0] + beta * coords_3d[1] + gamma * coords_3d[2]

                if ZBUFF: #zbuffer barely works
                    Z = surface_point[2]
                    if Z >= zbuffer[y, x]:
                        continue
                    zbuffer[y, x] = Z

                # finds normal of point based off of how far it is from vertices
                interpolated_normal = alpha * n1 + beta * n2 + gamma * n3
                interpolated_normal /= np.sqrt(np.dot(interpolated_normal, interpolated_normal))

                #direction from point to light
                light_dir = LIGHT_POS - surface_point
                light_dir /= np.sqrt(np.dot(light_dir, light_dir))

                #direction from point to camera
                view_dir = CAMERA_POS - surface_point
                view_dir /= np.sqrt(np.dot(view_dir, view_dir))

                #The diffuse lighting
                #for some reason using hte interpolated normal just makes it not work
                diffuse = (
                        d1 * alpha +
                        d2 * beta +
                        d3 * gamma
                )

                #what was bui tuong phong on about??
                #calculates unit reflection direction vector
                reflect_dir = 2.0 * np.dot(interpolated_normal, light_dir) * interpolated_normal - light_dir
                reflect_dir /= np.sqrt(np.dot(reflect_dir, reflect_dir))

                #cos of angle between direction of reflection and direction to camera
                spec_angle = max(0.0, np.dot(reflect_dir, view_dir))
                specular = REFLECTIVITY_CONSTANT * (spec_angle ** PHONG_EXPONENT)

                intensity = AMBIENT_INTENSITY + diffuse *255 + specular*255

                #gamma correction
                intensity = 1 * (intensity / 255.0) ** GAMMA

                #caps intensity just in case
                intensity = min(intensity, 1.0)

                color = np.array([base[2]*intensity,base[1]*intensity,base[0]*intensity])
                view[y, x] = color.astype(np.uint8)
    # print(colors)



@numba.njit()
def rasterize_phong(coords, view, zbuffer, av_normals, coords_3d, normal,color = (255,255,255), texturecoords=None):
    base = np.array(color, dtype=np.float64)
    red,green,blue = base
    A, B, C = coords.astype(np.float64)
    n1, n2, n3 = [np.asarray(n, dtype=np.float64) for n in av_normals]
    coords_3d = coords_3d.astype(np.float64)
    base = np.array(color,dtype=np.float64)
    #The 3d coordinated
    x1, y1, z1 = coords_3d[0]
    x2, y2, z2 = coords_3d[1]
    x3, y3, z3 = coords_3d[2]

    #Used for the bounding box
    min_x = max(int(min(A[0], B[0], C[0])), 0)
    max_x = min(int(max(A[0], B[0], C[0])) + 1, view.shape[1])
    min_y = max(int(min(A[1], B[1], C[1])), 0)
    max_y = min(int(max(A[1], B[1], C[1])) + 1, view.shape[0])

    #area of the triangle face
    v1, v2 = A - B, C - B
    total_area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

    if total_area == 0: #if triangle is too small, just skip the face
        return

    #denominator used to calculate for barycentric weight incrementation
    denom = (B[0] - C[0]) * (A[1] - C[1]) - (B[1] - C[1]) * (A[0] - C[0])

    #alpha and beta for the first (top left) point in bounding box
    x0,y0=min_x,min_y
    alpha_0 = ((B[0] - C[0]) * (y0 - C[1]) - (B[1] - C[1]) * (x0 - C[0])) / denom
    beta_0 = ((C[0] - A[0]) * (y0 - A[1]) - (C[1] - A[1]) * (x0 - A[0])) / denom

    #change in barycentric weights per change in x/y
    dalpha_x = -(B[1] - C[1]) / denom
    dalpha_y = (B[0] - C[0]) / denom
    dbeta_x = -(C[1] - A[1]) / denom
    dbeta_y = (C[0] - A[0]) / denom

    #diffuse lighting from all 3 average vertices calculated ahead of time
    d1 = max(n1.dot(LIGHT_VECTOR), 0)
    d2 = max(n2.dot(LIGHT_VECTOR), 0)
    d3 = max(n3.dot(LIGHT_VECTOR), 0)

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):

            #calculates 2d barycentric weights using previously found change values
            alpha = alpha_0 + (x - min_x) * dalpha_x + (y - min_y) * dalpha_y
            beta = beta_0 + (x - min_x) * dbeta_x + (y - min_y) * dbeta_y
            gamma = 1 - alpha - beta



            if alpha>=0 and beta>=0 and gamma>=0: #if the current point is in the triangle, continue
                surface_point = alpha * coords_3d[0] + beta * coords_3d[1] + gamma * coords_3d[2]

                if ZBUFF: #zbuffer barely works
                    Z = surface_point[2]
                    if Z >= zbuffer[y, x]:
                        continue
                    zbuffer[y, x] = Z

                # finds normal of point based off of how far it is from vertices
                interpolated_normal = alpha * n1 + beta * n2 + gamma * n3
                interpolated_normal /= np.sqrt(np.dot(interpolated_normal, interpolated_normal))

                #direction from point to light
                light_dir = LIGHT_POS - surface_point
                light_dir /= np.sqrt(np.dot(light_dir, light_dir))

                #direction from point to camera
                view_dir = CAMERA_POS - surface_point
                view_dir /= np.sqrt(np.dot(view_dir, view_dir))

                #The diffuse lighting
                #for some reason using hte interpolated normal just makes it not work
                diffuse = (
                        d1 * alpha +
                        d2 * beta +
                        d3 * gamma
                )

                #what was bui tuong phong on about??
                #calculates unit reflection direction vector
                reflect_dir = 2.0 * np.dot(interpolated_normal, light_dir) * interpolated_normal - light_dir
                reflect_dir /= np.sqrt(np.dot(reflect_dir, reflect_dir))

                #cos of angle between direction of reflection and direction to camera
                spec_angle = max(0.0, np.dot(reflect_dir, view_dir))
                specular = REFLECTIVITY_CONSTANT * (spec_angle ** PHONG_EXPONENT)

                intensity = AMBIENT_INTENSITY + diffuse *255 + specular*255

                #gamma correction
                intensity = 1 * (intensity / 255.0) ** GAMMA

                #caps intensity just in case
                intensity = min(intensity, 1.0)

                color = np.array([base[2]*intensity,base[1]*intensity,base[0]*intensity])
                view[y, x] = color.astype(np.uint8)
    # print(colors)