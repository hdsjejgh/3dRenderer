## 3d Renderer ##
This is a 3D Renderer made from scratch (no 3d graphics libraries, or graphics APIs)

The whole rendering pipeline is implemented from scratch (Model loading, transformations, per pixel shader calculation, rasterization). 
As of now, it has the ability to load OBJ files (as well as singular texture files), and perform a variety of transformations on the model (shifting, scaling, rotation, and general matrix transformation) and light source. 
The renderer has mouse controls, allowing zooming and rotating of the scene. Phong shading and (although now somewhat defunct) Gouraud shading are included. 

The renderer can handle a variety OBJ files (although larger files do not perform as well) as seen below.

### To Do ###

~~Migrate to pygame for rasterization~~

~~Add OBJ file support~~

~~Implement backface culling~~

~~Migrate to numpy~~

~~Implement a homemade rasterizer~~

~~Implement a smooth shading algorithm (Gouraud)~~

~~Implement matrix transformations~~

~~Add a Z Buffer~~

~~Refactor the messy code~~

Optimize the display more

~~Add Phong shading~~

~~Add texture loading~~

~~Add a mouse based movement system~~

Add nonlinear transformations

Add support for other file types

Support 4 dimensions (??)
# #
## Renderer in action ##
(Note, the renderer is regularly being updated and optimized, much more often than new videos are being added; therefore, these videos may represent an outdated version of the program)

### Simple 3d model with Phong Lighting ###
![](https://github.com/hdsjejgh/3dRenderer/blob/c4f40e6e428deed1e48a63de79caa2acc17bc139/media/phongkey.gif)
### Large 3d model with Phong Lighting ###
![](https://github.com/hdsjejgh/3dRenderer/blob/c4f40e6e428deed1e48a63de79caa2acc17bc139/media/hellkn.gif)
### Textured 3d model with Phong Lighting (beta) ###
![](https://github.com/hdsjejgh/3dRenderer/blob/c4f40e6e428deed1e48a63de79caa2acc17bc139/media/texturesworking.gif)


