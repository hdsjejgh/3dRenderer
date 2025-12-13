import parameters
import numba
import numpy as np

#lowkey forgot to put parameters. before every use of these so im just defining them like this here so its easier :wilted_rose:
FXAA_EDGE_THRESHOLD_MIN = parameters.FXAA_EDGE_THRESHOLD_MIN
FXAA_EDGE_THRESHOLD = parameters.FXAA_EDGE_THRESHOLD
FXAA_SUBPIX_TRIM = parameters.FXAA_SUBPIX_TRIM
FXAA_SUBPIX_TRIM_SCALE = parameters.FXAA_SUBPIX_TRIM_SCALE
FXAA_SUBPIX_CAP = parameters.FXAA_SUBPIX_CAP
FXAA_SEARCH_STEPS = parameters.FXAA_SEARCH_STEPS
FXAA_SEARCH_ACCELERATION = parameters.FXAA_SEARCH_ACCELERATION

#Anti-Aliasing


#Fast Approximate Anti-Aliasing
#(Based off the paper, credits to nvidia)
@numba.njit(parallel = True)
def FXAA(view,lum):
    #converted to [0,1] range
    view = view.astype(np.float32)/255
    #return vector
    outp = np.zeros(view.shape,dtype=np.float32)

    #viewlum is the luminance at each pixel
    lum = lum.astype(np.float32)
    viewLum = (view[:, :, 0] * lum[0] +
               view[:, :, 1] * lum[1] +
               view[:, :, 2] * lum[2])
    height,width = viewLum.shape

    #ignores edges
    for y in numba.prange(1,height-1):
        for x in range(1,width-1):
            #gets color and luminance from each of the 9 pixels near current pixel
            rgbN = view[y - 1, x]
            rgbS = view[y + 1, x]
            rgbE = view[y, x + 1]
            rgbW = view[y, x - 1]
            rgbM = view[y, x]
            rgbNW = view[y - 1, x - 1]
            rgbNE = view[y - 1, x + 1]
            rgbSW = view[y + 1, x - 1]
            rgbSE = view[y + 1, x + 1]

            lumaN = viewLum[y - 1, x]
            lumaS = viewLum[y + 1, x]
            lumaE = viewLum[y, x + 1]
            lumaW = viewLum[y, x - 1]
            lumaM = viewLum[y, x]
            lumaNW = viewLum[y - 1, x - 1]
            lumaNE = viewLum[y - 1, x + 1]
            lumaSW = viewLum[y + 1, x - 1]
            lumaSE = viewLum[y + 1, x + 1]

            #range of luminance within the 9 pixel region
            rangeMin = min(lumaN, min(lumaS, min(lumaE, min(lumaW, lumaM))))
            rangeMax = max(lumaN, max(lumaS, max(lumaE, max(lumaW, lumaM))))
            rnge = rangeMax-rangeMin
            #If the luminance does not vary much, skip
            if rnge < max(FXAA_EDGE_THRESHOLD_MIN, FXAA_EDGE_THRESHOLD*rangeMax):
                outp[y][x]=view[y][x]
                continue

            #average luminance of adjacent pixels
            lumaL = (lumaN+lumaW+lumaS+lumaE) * (1/4)
            #difference in luminance between pixel and its adjacents
            rangeL = abs(lumaL-lumaM)

            #how much to blend
            blendL = max(0.0, (rangeL-rnge) - FXAA_SUBPIX_TRIM) * FXAA_SUBPIX_TRIM_SCALE
            blendL = min(FXAA_SUBPIX_CAP, blendL)

            #average of surrounding 9 colors
            rgbL = rgbN + rgbNE + rgbNW + rgbS + rgbM + rgbE + rgbSE + rgbSW + rgbW
            rgbL *= (1.0/9.0)

            #how much of an edge there is vertically and horizomtally
            edgeVert = (abs((0.25 * lumaNW) + (-0.5 * lumaN) + (0.25 * lumaNE))
                        + abs((0.50 * lumaW) + (-1.0 * lumaM) + (0.50 * lumaE))
                        + abs((0.25 * lumaSW) + (-0.5 * lumaS) + (0.25 * lumaSE)))
            edgeHorz = (abs((0.25 * lumaNW) + (-0.5 * lumaW) + (0.25 * lumaSW))
                        + abs((0.50 * lumaN) + (-1.0 * lumaM) + (0.50 * lumaS))
                        + abs((0.25 * lumaNE) + (-0.5 * lumaE) + (0.25 * lumaSE)))
            #true = edge is horizontal, false = edge is vertical
            horzSpan = edgeHorz >= edgeVert

            if horzSpan:
                gradientN = abs(lumaN - lumaM)
                gradientS = abs(lumaS - lumaM)
                gradient = gradientN if gradientN > gradientS else gradientS

                #if edge is horizontal, searches vertically
                #starting spots for the search
                posN_x = x
                posN_y = y - 1
                posP_x = x
                posP_y = y + 1

                off_x = 0
                off_y = 1


            else:
                gradientW = abs(lumaW - lumaM)
                gradientE = abs(lumaE - lumaM)
                gradient = gradientW if gradientW > gradientE else gradientE

                # if edge is vertical, searches horizontally
                # starting spots for the search
                posN_x = x - 1
                posN_y = y
                posP_x = x + 1
                posP_y = y

                off_x = 1
                off_y = 0

            #luminance at end of negative and positive direction search
            lumaEndN = lumaM
            lumaEndP = lumaM
            doneN,doneP = False, False
            for i in range(FXAA_SEARCH_STEPS):
                if FXAA_SEARCH_ACCELERATION == 1:
                    if not doneN:
                        lumaEndN = viewLum[posN_y, posN_x]
                    if not doneP:
                        lumaEndP = viewLum[posP_y, posP_x]
                else:
                    raise Exception("Not implmented with FXAA_SEARCH_ACCELERATION>1 yet")

                #if search reached something too far from the initial, stop
                if abs(lumaEndN - lumaM) >= gradient:
                    doneN = True
                if abs(lumaEndP - lumaM) >= gradient:
                    doneP = True


                if doneN and doneP: break

                #changes position for next step
                if not doneN:
                    posN_x -= off_x
                    posN_y -= off_y
                if not doneP:
                    posP_x += off_x
                    posP_y += off_y

            #gets point in the middle of the search line
            indM_x = (posN_x + posP_x) // 2
            indM_y = (posN_y + posP_y) // 2

            rgbbM = view[indM_y, indM_x]
            finalColor = rgbbM * (1.0 - blendL) + rgbL * blendL
            outp[y,x]=finalColor
    #scales back to 0-255
    return outp