## The Greatest 3D Renderer To Ever Grace Technology ##
This is a 3D Renderer made from scratch (no 3d graphics libraries, or graphics APIs)

The whole rendering pipeline is implemented from scratch (Model loading, transformations, per pixel shader calculation, rasterization, etc.). 
As of now, it has the ability to load OBJ files (as well as singular texture files), load STL files, and perform a variety of transformations on the model (shifting, scaling, rotation, general matrix transformation, twisting, tapering) and light source. 
The renderer has mouse controls, allowing zooming and rotating of the scene. Phong shading and (although now somewhat defunct) Gouraud shading are included. 

The renderer can handle a variety of OBJ and STL files (although larger files do not perform as well) as seen below.

### How to use ###

(Quiet clunky as of now). 
Open the main.py file, and load the desired model into the Model variable with the correct class (different classes for different files are found in shapes.py). 
Around line 240, make sure the correct display function is being used, there is one for textured models and another for nontextured models. 
Add any pretransformations to the light or model to the pretransformation function. 
Add any per frams transformations in the Transformationloop function. 
Run the file. 
Drag and scroll to rotate and zoom.

### To Do ###

* ~~Migrate to pygame for rasterization~~

* ~~Add OBJ file support~~

* ~~Implement backface culling~~

* ~~Migrate to numpy~~

* ~~Implement a homemade rasterizer~~

* ~~Implement a smooth shading algorithm (Gouraud)~~
  * Not updated to handle the new system as of now

* ~~Implement matrix transformations~~

* ~~Add a Z Buffer~~

* ~~Refactor the messy code~~

* Optimize the display more
  * ~~Incremental barycentric coordinates~~ 

* ~~Add Phong shading~~

* ~~Add texture loading~~

* ~~Add a mouse based movement system~~

* Add nonlinear transformations
  * ~~Twisting~~
  * ~~Linear Tapering~~
  * Bending

* ~~Add real-time diagnostics~~
  * ~~EMA FPS tracker~~
  * ~~Face count~~

* Add more shading options

* Add Anti-Aliasing

* Add built in recording

* Add a real camera system

* Add support for other file types
  * STL Files
    * ~~ASCII based~~
    * Binary based

* Support 4 dimensions (??)
# #
## Renderer in action ##
(Note, the renderer is regularly being updated and optimized, much more often than new videos are being added; therefore, these videos may represent an outdated version of the program)

As of now the media is **NOT** up to date

### Simple 3d model with Phong Lighting \[968 faces\] ###
![](https://github.com/hdsjejgh/3dRenderer/blob/c4f40e6e428deed1e48a63de79caa2acc17bc139/media/phongkey.gif)
### Large 3d model with Phong Lighting \[19381 faces\] ###
![](https://github.com/hdsjejgh/3dRenderer/blob/c4f40e6e428deed1e48a63de79caa2acc17bc139/media/hellkn.gif)
### Textured 3d model with Phong Lighting (beta) \[284 faces\] ###
![](https://github.com/hdsjejgh/3dRenderer/blob/c4f40e6e428deed1e48a63de79caa2acc17bc139/media/texturesworking.gif)


