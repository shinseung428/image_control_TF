# Different image controls in Tensorflow

## Implemented Functions  
* Block Pixels  
* Conv Noise  
* Block Patch(fixed size)  
* Block Patch(random size)  
* Keep Patch

## Block Pixels  
Randomly blocks each pixel to 0 with probability p. 
![Alt text](images/blockpixels.jpg?raw=true "blockpixels")  

## Conv Noise
Uses gaussian kernel to blur images and adds additional noise to each pixels.  
![Alt text](images/convnoise.jpg?raw=true "convnoise")  

## Block Patch  
Randomly blocks p x p patched region in the image.  
The size of the patch can be changed.  
![Alt text](images/blockpatch.jpg?raw=true "blockpatch")  

## Block Patch(Random Size)  
Randomly blocks patched region in the image.  
The size of the patch is random.  
![Alt text](images/blockpatchrand.jpg?raw=true "blockpatchrand")  

## Keep Patch  
All pixels outside randomly chosen p x p patch are set to zero.  
![Alt text](images/keeppatch.jpg?raw=true "keeppatch")  