# Comic Generator using Stable Diffusion 
This is an application that you can run in an IDE to create custom comics using Stable Diffusion and pretrained model provaided by ogkalu.

## Features
- __text2img__ Stable Diffusion pipeline for generating images based on positive and negative prompt.
- __img2img__ Stable Diffusion pipeline for generating images based on your simple painting as well as additional prompt to describe it.
- __inpainting__  Stable Diffusion pipeline for mostly adding and fixing elements of generated image (paint and prompt part of image to fix).
- __costum comic artstyle trained model__ provided by the user ogkalu on Huggingface.

## Installation

Since everything works locally you have to firstly download [inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting/tree/main) and [pretrained comic model](https://huggingface.co/ogkalu/Comic-Diffusion) for stable diffusion.

In main file you then have to set path at line 36: 

```
pipelinePaint = StableDiffusionInpaintPipeline.from_pretrained(
        r"inpainintg_parent_folder", revision="fp16",
        torch_dtype=torch.float16)
```


and in modelID file set the comic path (example): 

```
modelID = r"C:\Users\User\.cache\huggingface\diffusers\models--ogkalu--Comic-Diffusion\snapshots\b0f8b7655c342796dd3e25e2182f200919619e7c" 
```

Finally copy the unet folder from Comic to inpainting (replace). 

Note that at least 8GB of VRAM is needed (supports only Nividia Cuda GPU) as well as torch compiled with cuda! 

# Aplication

Most things should be self explanatory. Know limitations are from pretrained model as well as consisterncy of charachters. I recommend prompting with known people like Jack Sparrow or John Wick to get some consistency. Comic is of size 3x4 (12 pictures) and can be exported as .pdf and printed in A4 format.

Look for inspiration on [OpenArt](https://openart.ai/) for better prompting. 

# Results: 
### img2img:
Example use of simple drawing. 

![img3](https://user-images.githubusercontent.com/122792037/212675369-4e3ea7bc-d40e-40c1-b7f5-999a325b40c2.png)

### Inpainting to fix/add elements to generated image: 
As you can see from the image I have replaced the blue soldier's head as well as added blood to the body below. 

![fig5](https://user-images.githubusercontent.com/122792037/212675415-73d9874e-2696-42c9-9961-d87c11eb6d64.png)

### Consistency of charachters: 
Consistency of characters is achieved through proper prompting (i.e. by utilizing famous people and regenerating images with different prompts). If you want custom characters then additional training is needed which is currently not supported. 

![gifg](https://user-images.githubusercontent.com/122792037/212675451-5e00b1c9-8b44-4b82-b110-c212e82d4390.jpg)

### Generated comic in sloveninan: 
[GeneratedComic.pdf](https://github.com/Friday202/ComicGenerator/files/10425643/GeneratedComic.pdf)

### Note
I am well aware that the application is not editor friendly for any modifications as it is all in one file. 
