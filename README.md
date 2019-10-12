# Image generation with Tensorflow 2 and GANs
This project's goal is to **generate images** using **existing images** like an artist.  
Tested with **Tensorflow 2** and **GANs (Generative Adversarial Networks)**, (1080x773px, 3 colors) images on Nvidia 2080 GPU, VScode.

# How to use it
Copy all pictures you want to use in the **input** directory of this project (tested with pictures of the same size : 1080x773px, 3 colors).  
Change parameters in main.py and GAN.py if needed (imageHeight, imageWidth, redimRatio, dpi, percentageOfImagesToKeep, imagesPerIteration, batch_size, sample_interval, epochs).  
Run main.py.  

# The result
In the **resized** directory : generate 4 resized images.  

In the **output** directory :  
- Generate 1 image for every sample_interval epochs.  
- Generate 1 image called output.png wich name never changes, so you can see it in live.  
- Generate 1 GIF after all epochs.

# Parameters to change
**percentageOfImagesToKeep** : Percentage of images to keep in the input directory. If 20 and if you have 3000 images, it 'll keep 20% of 3000 images : 600.  
**imagesPerIteration** : 3 will generate 1 big image containing 3x3 pictures per iteration (so 9 pictures).  
**sample_interval** : 20 means generate an output image every 20 epochs.  
**batch_size** : Batch size in the main GAN loop. Bigger means faster but it may give bad results.  
**epochs** : Number of epoch in the main GAN loop. I observe Better results when the number of epoch is not that big (50 to 400). 

**imageHeight** : Change it one time, the pixel height.  
**imageWidth** : Change it one time, the pixel width.  
**redimRatio** : 4 means the resized image will be (imageHeight / 4, imageWidth / 4). Used to reduce memory and go faster.  
**dpi** : the size of the final picture. Higher means better image quality but it takes longer to create the files.  
