## 3d Renderer ##
This is a 3d renderer made fully from scratch in Python

The entire 3d rendering pipeline is implemented. It loads OBJ files, performs transformations, applies shaders (right now Gouraud, Lambertian, and a distance based shader are implemented), and rasterizes a display. It is decently optimized thanks to numpy and numba and can handle decently large OBJ files as shown below

### To Do ###

~~Add OBJ file support~~

~~Implement backface culling~~

~~Migrate to numpy~~

~~Implement a homemade rasterizer~~

~~Implement a smooth shading algorithm~~

~~Add matrix transformations~~

~~Add a Z Buffer~~

~~Refactor the messy code (all of it)~~

Optimize the display more

~~Add Phong shading~~

~~Add texture loading~~

~~Add a mouse based movement system~~

Add support for other file types

Support 4 dimensions (??)
# #
### Simple 3d model with Phong Lighting ###
![](https://github.com/hdsjejgh/3dRenderer/blob/c4f40e6e428deed1e48a63de79caa2acc17bc139/media/phongkey.gif)
### Large 3d model with Phong Lighting ###
![](https://github.com/hdsjejgh/3dRenderer/blob/c4f40e6e428deed1e48a63de79caa2acc17bc139/media/hellkn.gif)
### Textured 3d model with Phong Lighting (beta) ###
![](https://github.com/hdsjejgh/3dRenderer/blob/c4f40e6e428deed1e48a63de79caa2acc17bc139/media/texturesworking.gif)


