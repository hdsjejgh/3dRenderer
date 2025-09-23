import random
import parameters
import numpy as np
import numba
import math
#-----------------------------------------------#

#Defunct Fragment Shaders

#Initializes random colors for the randomShader
COLORS = tuple(tuple(random.randint(0,255) for ii in range(3)) for i in range(100))
#Assigns random color to each face based on its id
#Now defunct
def randomShader():
    def wrapper(face):
        return COLORS[face.id%len(COLORS)]
    return wrapper

#Shader that mades faces darker furter away, and lighter closer
#Now defunct
def distShader(mult=0.05,base=(255,255,255)): #Mult controls steepness of sigmoid function
    #steeper Mult = steeper sigmoid = quicker dropoff of brightness
    def wrapper(face):
        #Bastardized version of the standard sigmoid function
        val = 255 * (1-  (1/(1+math.exp(-mult*face.z)))   )
        val = min(255,val)

        return tuple(map(lambda x:x*(val/255),base))
    return wrapper

#Lambertian Shader
#Technically not Lambertian, but its close enough
#Now defunct
def Lambertian():
    def wrapper(face):
        return np.array([205*(np.dot(parameters.VIEW_VECTOR,face.normal))**2+50]*3, dtype=np.uint8)
    return wrapper

#Gouraud shader
#Now defunct
def Gouraud():
    def wrapper(point,params,points3d,normals):
        dists = np.linalg.norm(points3d - point, axis=1)
        colors = [np.array([205*(np.dot(parameters.VIEW_VECTOR,normal))**2+50]*3, dtype=np.uint8) for normal in normals]
        color = sum(colors[i]/dists[i] for i in range(3))*np.mean(dists)
        return color
    return wrapper

#Primitive rasterizer
#Now defunct
@numba.njit()
def rasterize(coords, color, view):
    #2d Coordinates of vertices
    A, B, C = coords

    #Used for the bounding box
    min_x = max(int(min(A[0], B[0], C[0])), 0)
    max_x = min(int(max(A[0], B[0], C[0])) + 1, view.shape[1])
    min_y = max(int(min(A[1], B[1], C[1])), 0)
    max_y = min(int(max(A[1], B[1], C[1])) + 1, view.shape[0])

    #area of the 2d triangle
    area = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
    if area == 0:
        return

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            #berycentric weights based on 2d projection
            w0 = (B[0] - A[0]) * (y - A[1]) - (B[1] - A[1]) * (x - A[0])
            w1 = (C[0] - B[0]) * (y - B[1]) - (C[1] - B[1]) * (x - B[0])
            w2 = (A[0] - C[0]) * (y - C[1]) - (A[1] - C[1]) * (x - C[0])

            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                #If point in triangle
                view[y, x] = color


#Operational Fragment Shaders

#Textured Phong shader
#Has to be separated from nontextured phong because numba likes being difficult
@numba.njit() #Numba used to provide massive speedups
def rasterize_phong_texture(coords, view, zbuffer, av_normals, coords_3d, texturecoords=None, texture=np.empty((1,1),np.int64),LIGHT_POS=parameters.LIGHT_POS, LIGHT_VECTOR=parameters.LIGHT_VECTOR):
    #All 3 2 dimensions coordinates
    A, B, C = coords.astype(np.float64)
    #Vertex normals of the 3 vertices
    n1, n2, n3 = [np.asarray(n, dtype=np.float64) for n in av_normals]
    #3 dimensional coordinates
    coords_3d = coords_3d.astype(np.float64)

    #left,right,down,up
    #Used for the bounding box
    min_x = max(int(min(A[0], B[0], C[0])), 0)
    max_x = min(int(max(A[0], B[0], C[0])) + 1, view.shape[1])
    min_y = max(int(min(A[1], B[1], C[1])), 0)
    max_y = min(int(max(A[1], B[1], C[1])) + 1, view.shape[0])

    #area of the triangle face in 2 dimensions
    v1, v2 = A - B, C - B
    total_area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

    #Rejects the face if its area was calculated too small
    if total_area <= 0:
        return

    #Denominator used to calculate for barycentric weight incrementation
    denom = (B[0] - C[0]) * (A[1] - C[1]) - (B[1] - C[1]) * (A[0] - C[0])

    #Alpha and Beta for the first (top left) point in the bounding box
    #Gamma is just calculated as 1-alpha-beta later so its not calculated here
    x0,y0=min_x,min_y
    alpha_0 = ((B[0] - C[0]) * (y0 - C[1]) - (B[1] - C[1]) * (x0 - C[0])) / denom
    beta_0 = ((C[0] - A[0]) * (y0 - A[1]) - (C[1] - A[1]) * (x0 - A[0])) / denom

    #Change in barycentric weights per change in x/y
    #Gamma is just calculated as 1-alpha-beta later so its not calculated here
    dalpha_x = -(B[1] - C[1]) / denom
    dalpha_y = (B[0] - C[0]) / denom
    dbeta_x = -(C[1] - A[1]) / denom
    dbeta_y = (C[0] - A[0]) / denom

    # Base alpha and beta values
    # Gets incremented before every change in x and y
    # Subtracts the changes at first so i can have the incrementation at the start of the loop instead of the end
    alpharow = alpha_0 - dalpha_x - dalpha_y
    betarow = beta_0 - dbeta_x - dbeta_y

    for y in range(min_y, max_y): #Iterates over y coordinates top to bottm

        # Increments base alphas and betas by y change
        alpharow += dalpha_y
        betarow += dbeta_y

        alpha = alpharow
        beta = betarow

        for x in range(min_x, max_x): #Iterates over x coordinates left to right

            # Increments base alphas and betas by x change
            alpha += dalpha_x
            beta += dbeta_x
            # Calculates gamma based on alpha and beta
            gamma = 1 - alpha - beta

            if alpha>=0 and beta>=0 and gamma>=0: #if the current point is in the triangle, continue

                if texture.size != 0:  # "If texture exists"

                    # Texture coordinate found based on barycentric weights
                    tc = alpha * texturecoords[0] + beta * texturecoords[1] + gamma * texturecoords[2]

                    # Texture coordinate is rounded, 0 indexed, and bounded
                    i = int(round(tc[1])) - 1
                    j = int(round(tc[0])) - 1
                    i = max(0, min(texture.shape[0] - 1, i))
                    j = max(0, min(texture.shape[1] - 1, j))

                    # Base represents the color of the model at the current pixel
                    base = np.asarray(texture[i, j], dtype=np.float64)

                #Finds the 2d coordinate projected onto the 3d face
                surface_point = alpha * coords_3d[0] + beta * coords_3d[1] + gamma * coords_3d[2]

                if parameters.ZBUFF: #Zbuffers if enabled
                    #Skips point if closer point already drawn over it
                    Z = surface_point[2]
                    if Z >= zbuffer[y, x]:
                        continue
                    zbuffer[y, x] = Z

                # Finds normal of point based off of how far it is from vertices
                interpolated_normal = alpha * n1 + beta * n2 + gamma * n3
                interpolated_normal /= np.sqrt(np.dot(interpolated_normal, interpolated_normal))

                #Direction from the point to the global light source
                light_dir = LIGHT_POS - surface_point
                light_dir /= np.sqrt(np.dot(light_dir, light_dir))

                #Direction from the point to global camera position
                view_dir = parameters.CAMERA_POS - surface_point
                view_dir /= np.sqrt(np.dot(view_dir, view_dir))

                # The Diffuse lighting
                #While i was cleaning up i tried again and it works just fine now???
                diffuse = max(interpolated_normal.dot(-light_dir), 0)

                #(What was bui tuong phong on about??)
                #Calculates reflection direction vector
                reflect_dir = 2.0 * np.dot(interpolated_normal, light_dir) * interpolated_normal - light_dir
                reflect_dir /= np.sqrt(np.dot(reflect_dir, reflect_dir))

                #Cos of angle between direction of reflection and direction to camera
                spec_angle = max(0.0, np.dot(reflect_dir, view_dir))

                #Specular highlight calculated
                specular = parameters.REFLECTIVITY_CONSTANT * (spec_angle ** parameters.PHONG_EXPONENT)

                #Final lighting intensity
                intensity = parameters.AMBIENT_INTENSITY + diffuse *255 + specular*255

                #Gamma correction
                #Also puts intensity in unit range
                intensity = (intensity / 255.0) ** parameters.GAMMA
                intensity = min(intensity, 1.0)

                #Final color calculated based on the color of pixel and intensity of pixel
                #The color of the model is converted to BGR from RGB as well
                color = np.array([base[2]*intensity,base[1]*intensity,base[0]*intensity])
                view[y, x] = color.astype(np.uint8)

#Nontextured Phong shader
#Has to be separated from textured phong because numba likes being difficult
@numba.njit()
def rasterize_phong(coords, view, zbuffer, av_normals, coords_3d, color = (255,255,255),LIGHT_POS=parameters.LIGHT_POS, LIGHT_VECTOR=parameters.LIGHT_VECTOR):

    # All 3 2 dimensions coordinates
    A, B, C = coords.astype(np.float64)
    # Vertex normals of the 3 vertices
    n1, n2, n3 = [np.asarray(n, dtype=np.float64) for n in av_normals]
    # 3 dimensional coordinates
    coords_3d = coords_3d.astype(np.float64)
    #Base color of the model
    #White by default
    base=color

    # left,right,down,up
    # Used for the bounding box
    min_x = max(int(min(A[0], B[0], C[0])), 0)
    max_x = min(int(max(A[0], B[0], C[0])) + 1, view.shape[1])
    min_y = max(int(min(A[1], B[1], C[1])), 0)
    max_y = min(int(max(A[1], B[1], C[1])) + 1, view.shape[0])

    # area of the triangle face in 2 dimensions
    v1, v2 = A - B, C - B
    total_area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

    # Rejects the face if its area was calculated too small
    if total_area <= 0:
        return

    # Denominator used to calculate for barycentric weight incrementation
    denom = (B[0] - C[0]) * (A[1] - C[1]) - (B[1] - C[1]) * (A[0] - C[0])

    # Alpha and Beta for the first (top left) point in the bounding box
    # Gamma is just calculated as 1-alpha-beta later so its not calculated here
    x0, y0 = min_x, min_y
    alpha_0 = ((B[0] - C[0]) * (y0 - C[1]) - (B[1] - C[1]) * (x0 - C[0])) / denom
    beta_0 = ((C[0] - A[0]) * (y0 - A[1]) - (C[1] - A[1]) * (x0 - A[0])) / denom

    # Change in barycentric weights per change in x/y
    # Gamma is just calculated as 1-alpha-beta later so its not calculated here
    dalpha_x = -(B[1] - C[1]) / denom
    dalpha_y = (B[0] - C[0]) / denom
    dbeta_x = -(C[1] - A[1]) / denom
    dbeta_y = (C[0] - A[0]) / denom

    #Base alpha and beta values
    #Gets incremented before every change in x and y
    #Subtracts the changes at first so i can have the incrementation at the start of the loop instead of the end
    alpharow = alpha_0-dalpha_x-dalpha_y
    betarow = beta_0-dbeta_x-dbeta_y

    for y in range(min_y, max_y):  # Iterates over y coordinates top to bottm

        #Increments base alphas and betas by y change
        alpharow += dalpha_y
        betarow += dbeta_y

        alpha = alpharow
        beta = betarow

        for x in range(min_x, max_x):  # Iterates over x coordinates left to right

            #Increments base alphas and betas by x change
            alpha+=dalpha_x
            beta+=dbeta_x
            #Calculates gamma based on alpha and beta
            gamma = 1 - alpha - beta

            if alpha >= 0 and beta >= 0 and gamma >= 0:  # if the current point is in the triangle, continue

                # Finds the 2d coordinate projected onto the 3d face
                surface_point = alpha * coords_3d[0] + beta * coords_3d[1] + gamma * coords_3d[2]

                if parameters.ZBUFF:  # Zbuffers if enabled
                    # Skips point if closer point already drawn over it
                    Z = surface_point[2]
                    if Z >= zbuffer[y, x]:
                        continue
                    zbuffer[y, x] = Z

                # Finds normal of point based off of how far it is from vertices
                interpolated_normal = alpha * n1 + beta * n2 + gamma * n3
                interpolated_normal /= np.sqrt(np.dot(interpolated_normal, interpolated_normal))

                # Direction from the point to the global light source
                light_dir = LIGHT_POS - surface_point
                light_dir /= np.sqrt(np.dot(light_dir, light_dir))

                # Direction from the point to global camera position
                view_dir = parameters.CAMERA_POS - surface_point
                view_dir /= np.sqrt(np.dot(view_dir, view_dir))

                # The Diffuse lighting
                # While i was cleaning up i tried again and it works just fine now???
                diffuse = max(interpolated_normal.dot(-light_dir), 0)

                # (What was bui tuong phong on about??)
                # Calculates reflection direction vector
                reflect_dir = 2.0 * np.dot(interpolated_normal, light_dir) * interpolated_normal - light_dir
                reflect_dir /= np.sqrt(np.dot(reflect_dir, reflect_dir))

                # Cos of angle between direction of reflection and direction to camera
                spec_angle = max(0.0, np.dot(reflect_dir, view_dir))

                # Specular highlight calculated
                specular = parameters.REFLECTIVITY_CONSTANT * (spec_angle ** parameters.PHONG_EXPONENT)

                # Final lighting intensity
                intensity = parameters.AMBIENT_INTENSITY + diffuse * 255 + specular * 255

                # Gamma correction
                # Also puts intensity in unit range
                intensity = (intensity / 255.0) ** parameters.GAMMA
                intensity = min(intensity, 1.0)

                # Final color calculated based on the color of pixel and intensity of pixel
                # The color of the model is converted to BGR from RGB as well
                color = np.array([base[2] * intensity, base[1] * intensity, base[0] * intensity])
                view[y, x] = color.astype(np.uint8)



