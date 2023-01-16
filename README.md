# ComicGenerator

This is an application that you can run in an IDE like PyCharm to create custom comics based off Stable Diffusion. 

# Installation instructions:

Since everything works locally you have to firstly download inapinting and custom comic model stable diffusion, avabile at: 

https://huggingface.co/runwayml/stable-diffusion-inpainting/tree/main

and

https://huggingface.co/ogkalu/Comic-Diffusion

In main file you then have to set path at line 36: 

pipelinePaint = StableDiffusionInpaintPipeline.from_pretrained(
        r"inpainintg_parent_folder", revision="fp16",
        torch_dtype=torch.float16)
  


and in modelID file set the comic path (example): 

modelID = r"C:\Users\User\.cache\huggingface\diffusers\models--ogkalu--Comic-Diffusion\snapshots\b0f8b7655c342796dd3e25e2182f200919619e7c" 


Finally copy the unet folder from Comic to inpainting (replace). 

At least 8GB of VRAM is needed (supports only Nividia Cuda GPU) as well as torch compiled with cuda! 

# Aplication

Most things should be self explanatory. Know limitations are from pretrained model as well as consisterncy of charachters. I recommend prompting with known people like Jack Sparrow or John Wick to get some consistency. Comic is of size 3x4 (12 pictures) and can be exported as .pdf and printed. 

Look for inspiration on https://openart.ai/ for better prompting. 

# Results: 
img2img:
![img3](https://user-images.githubusercontent.com/122792037/212675369-4e3ea7bc-d40e-40c1-b7f5-999a325b40c2.png)

inpainting to fix/add elements: 

![fig5](https://user-images.githubusercontent.com/122792037/212675415-73d9874e-2696-42c9-9961-d87c11eb6d64.png)

Consistency of charachters: 
![gifg](https://user-images.githubusercontent.com/122792037/212675451-5e00b1c9-8b44-4b82-b110-c212e82d4390.jpg)

Generated comic in sloveninan: 
[GeneratedComic.pdf](https://github.com/Friday202/ComicGenerator/files/10425643/GeneratedComic.pdf)

