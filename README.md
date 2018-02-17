# Different image controls in Tensorflow



## Block Pixels  
This function randomly blocks each pixel to 0 with probability p. 
![Alt text](images/blockpixels.jpg?raw=true "blockpixels")  

## Conv Noise
This function uses gaussian kernel to blur images and adds additional noise to each pixels.  
![Alt text](images/convnoise.jpg?raw=true "convnoise")  

## Block Patch  
This function randomly blocks patched region in the image.  
The size of the patch can be changed.  
![Alt text](images/blockpatch.jpg?raw=true "blockpatch")  

## Block Patch(Random Size)  
This function randomly blocks patched region in the image.  
The size of the patch is random.  

## Keep Patch  
All pixels outside randomly chosen k x k patch are set to zero.  
![Alt text](images/keeppatch.jpg?raw=true "keeppatch")  