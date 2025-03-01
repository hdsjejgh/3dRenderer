FOV = 30
coords = [
    [-2,2,2],[2,2,2],[2,-2,2],[-2,-2,2],
    [-2,2,6],[2,2,6],[2,-2,6],[-2,-2,6],
]
TwoDimensionalCoords = [(i[0]*FOV/(i[2]+FOV),i[1]*FOV/(i[2]+FOV)) for i in coords]
#print(TwoDimensionalCoords)
faces = [
    [0,1,2,3],[4,5,6,7],[0,4,3,7],[1,5,2,6],[0,1,4,5],[2,3,7,6],
]
symbols = list('123456')
avZ = [sum(coords[ii][2] for ii in i)/4 for i in faces]

def triArea(x1,y1,x2,y2,x3,y3):
    return (1 / 2) * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def quadArea(x1,y1,x2,y2,x3,y3,x4,y4):
    return 1/2 * abs((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (x2*y1 + x3*y2 + x4*y3 + x1*y4))



for y in (i * 0.3 for i in range(-10,10)):
    disp=""
    for x in (i * 0.3 for i in range(-10,10)):
        front = []
        for idx,face in enumerate(faces):
            sum = triArea(x,y,TwoDimensionalCoords[face[0]][0],TwoDimensionalCoords[face[0]][1],TwoDimensionalCoords[face[-1]][0],TwoDimensionalCoords[face[-1]][1])
            for i in range(3):
                x1=x
                y1=y
                x2=TwoDimensionalCoords[face[i]][0]
                y2=TwoDimensionalCoords[face[i]][1]
                x3 = TwoDimensionalCoords[face[i+1]][0]
                y3 = TwoDimensionalCoords[face[i+1]][1]
                sum+=triArea(x1,y1,x2,y2,x3,y3)

            x1,y1 = TwoDimensionalCoords[face[0]]
            x2, y2 = TwoDimensionalCoords[face[1]]
            x3, y3 = TwoDimensionalCoords[face[2]]
            x4, y4 = TwoDimensionalCoords[face[3]]
            area = quadArea(x1,y1,x2,y2,x3,y3,x4,y4)

            #print(f"({x,y}) {area=} {sum=} {idx=}")
            if round(area,5)==round(sum,5):
                front.append((idx,avZ[idx]))
        if len(front)==0:
            disp+="  "
            continue
        front.sort()
        disp+=str(front[0][0])+' '
    print(disp)


