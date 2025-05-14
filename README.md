## 3d Renderer ##
This is a 3d renderer made fully from scratch in Python

The entire 3d rendering pipeline is implemented. It loads OBJ files, performs transformations, applies shaders (right now Gouraud, Lambertian, and a distance based shader are implemented), and rasterizes a display. It is decently optimized thanks to numpy and numba and can handle decently large OBJ files as shown below

### To Do ###

~~Add OBJ file support~~

~~Implement backface culling~~

~~Migrate to numpy~~

~~Implement a homemade rasterizer~~

~~Implement a smooth shading algorithm~~

Add matrix transformations

Refactor the messy code (all of it)

Optimize the display

Add Phong shading

Add texture loading

Add support for other file types

Support 4 dimensions
# #
### Simple 3d model display ###
![](https://github.com/hdsjejgh/3dRenderer/blob/136cd10758a078b865bb4f9022f190069a261a52/media/mokey.gif)
### Large 3d model display ###
![](https://github.com/hdsjejgh/3dRenderer/blob/136cd10758a078b865bb4f9022f190069a261a52/media/hekewjigwke.gif)
### Smooth Gouraud Shading (Beta) ###
![](https://github.com/hdsjejgh/3dRenderer/blob/136cd10758a078b865bb4f9022f190069a261a52/media/gourd.gif)
