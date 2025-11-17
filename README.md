## The Greatest 3D Renderer To Ever Grace Technology ##
This is a 3D Renderer made from scratch (no 3d graphics libraries, or graphics APIs)

The whole rendering pipeline is implemented from scratch (Model loading, transformations, per pixel shader calculation, rasterization, etc.). 
As of now, it has the ability to load OBJ files (as well as singular texture files), load STL files, and perform a variety of transformations on the model (shifting, scaling, rotation, general matrix transformation, twisting, tapering) and light source. 
The renderer has mouse controls, allowing zooming and rotating of the scene. Phong shading, Gouraud shading and Lambertian shading are included. 

The renderer can handle a variety of OBJ and STL files (although larger files do not perform as well) as seen below.

### How to use ###

(Quiet clunky as of now). 
Open the main.py file, and load the desired model into the Model variable with the correct class (different classes for different files are found in shapes.py). 
Choose the correct shader function from the displayFunctions.py file
Add any pretransformations to the light (found ing lightFunction.py) or model (found under the File class in shapes.py) to the pretransformation function. 
Add any per frame transformations in the Transformationloop function. 
Run the file. 
Drag and scroll to rotate and zoom.

### To Do ###

* ~~Migrate to pygame for rasterization~~

* ~~Add OBJ file support~~

* ~~Implement backface culling~~

* ~~Migrate to numpy~~

* ~~Implement a homemade rasterizer~~

* ~~Implement a smooth shading algorithm (Gouraud)~~

* ~~Implement matrix transformations~~

* ~~Add a Z Buffer~~

* ~~Refactor the messy code~~

* ~~Optimize the display more~~
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
  * ~~Phong~~
  * ~~Gouruad~~
  * ~~Lambertian~~ 

* ~~Add Anti-Aliasing~~
  * ~~FXAA~~

* Add built in recording

* Add a real camera system

* Add support for other file types
  * ~~STL Files~~
    * ~~ASCII based~~
    * ~~Binary based~~

* Support 4 dimensions (??)
# #
## Renderer in action ##
(Note, the renderer is regularly being updated and optimized, much more often than new videos are being added; therefore, these videos may represent an outdated version of the program)

As of now the media is up to date

### Larger 3d model with Lambertian Lighting \[968 faces\] ###
![](https://github.com/hdsjejgh/3dRenderer/blob/399d2f5bbcde72472cc075782027dcf5e0127442/media/lamb.gif)
### Textured 3d model with Gouraud Lighting \[408 faces\] ###
![](https://github.com/hdsjejgh/3dRenderer/blob/399d2f5bbcde72472cc075782027dcf5e0127442/media/gouraudtexture.gif)
### Textured 3d model with Phong Lighting \[284 faces\] ###
![](https://github.com/hdsjejgh/3dRenderer/blob/399d2f5bbcde72472cc075782027dcf5e0127442/media/phongtextured.gif)

# #
### Nonlinear transformation (Twist) ###
![](https://github.com/hdsjejgh/3dRenderer/blob/399d2f5bbcde72472cc075782027dcf5e0127442/media/twist.gif)
