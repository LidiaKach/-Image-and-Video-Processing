**_Week 1**_ 

1. Write  a computer program capable of reducing the number of intensity levels  in an image from 256 to 2, in integer 
powers of 2. The desired number of  intensity levels needs to be a variable input to your program.

2. Using  any programming language you feel comfortable with (it is though  recommended to use the provided free Matlab),
load an image and then  perform a simple spatial 3x3 average of image pixels. In other words,  replace the value of
every pixel by the average of the values in its 3x3  neighborhood. If the pixel is located at (0,0), this means 
averaging  the values of the pixels at the positions (-1,1), (0,1), (1,1), (-1,0),  (0,0), (1,0), (-1,-1), (0,-1),
and (1,-1). Be careful with pixels at the  image boundaries. Repeat the process for a 10x10 neighborhood and again
for a 20x20 neighborhood. Observe what happens to the image (we will  discuss this in more details in the very 
near future, about week 3).

3. Rotate the image by 45 and 90 degrees (Matlab provides simple command lines for doing this).

4. For  every 3 \times 33×3 block of the image (without overlapping), replace  all corresponding 9 pixels by their 
average. This operation simulates  reducing the image spatial resolution. Repeat this for 5 \times 55×5  blocks and 
 7 \times 77×7 blocks. If you are using Matlab, investigate  simple command lines to do this important operation.
 
 
 **_Week 2**_ 
 
Do a basic implementation of JPEG:
 
Divide the image into non-overlapping 8x8 blocks.
Compute the DCT (discrete cosine transform) of each block. This is implemented in popular packages such as Matlab.
Quantize each block. You can do this using the tables in the video or simply divide each coefficient by N, round 
the result to the nearest integer, and multiply back by N. Try for different values of N.
You can also try preserving the 8 largest coefficients (out of the total of 8x8=64), and simply rounding 
cd..them to the closest integer.
Visualize the results after inverting the quantization and the DCT.

Repeat the above but instead of using the DCT, use the FFT (Fast Fourier Transform).

Repeat the above JPEG-type compression but don’t use any transform, simply perform quantization on the original image.

Do JPEG now for color images. In Matlab, use the rgb2ycbcr command to convert the Red-Green-Blue image to a Lumina and Chroma one; then perform the JPEG-style compression on each one of the three channels independently. After inverting the compression, invert the color transform and visualize the result. While keeping the compression ratio constant for the Y channel, increase the compression of the two chrominance channels and observe the results.

Compute the histogram of a given image and of its prediction errors. If the
pixel being processed is at coordinate (0,0), consider
predicting based on just the pixel at (-1,0);
predicting based on just the pixel at (0,1);
predicting based on the average of the pixels at (-1,0), (-1,1), and (0,1).
Compute the entropy for each one of the predictors in the previous exercise. Which predictor will compress better?
