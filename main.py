import tkinter as tk
import tkinter
import tkinter.tix
import tkinter.ttk
from tkinter import *

from tkinter import filedialog
import io
import random
import PIL.ImageOps
import customtkinter as ctk

from PIL import ImageTk, ImageGrab, ImageDraw2
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel

from model import modelID

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer



# Load models:

def loadPaintingSD():
    global pipelinePaint
    pipelinePaint = StableDiffusionInpaintPipeline.from_pretrained(
        r"PATH_TO_INPAINT_MODEL", revision="fp16",
        torch_dtype=torch.float16)
    pipelinePaint = pipelinePaint.to(device)


def loadComicSD():
    global pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(modelID, revision="fp16", torch_dtype=torch.float16)
    pipeline = pipeline.to(device)

def loadImg2ImgSD():
    global pipelineImg2Img
    pipelineImg2Img = StableDiffusionImg2ImgPipeline.from_pretrained(modelID, revision="fp16",  torch_dtype=torch.float16)
    pipelineImg2Img = pipelineImg2Img.to(device)

#####################################################################################################################

def getFirstCoordinates(event):
    boxCoordinates.clear()
    boxCoordinates.append((event.x, event.y))


def getLastCoordinates(event):
    boxCoordinates.append((event.x, event.y))
    drawMessageBox()
    print(boxCoordinates)


def drawMessageBox():
    global boxCoordinates
    drawMessage = ImageDraw.Draw(image3)
    drawMessage.ellipse(boxCoordinates, fill="white", outline="black")
    updateImgSlot(3)


def get_x_and_y(event):
    global lasx, lasy, color
    lasx, lasy = event.x, event.y
    brushSize = brushSlider.get()
    drawMask.ellipse([lasx - brushSize, lasy - brushSize, lasx + brushSize, lasy + brushSize], fill=color)

    updateImgSlot(2)


def draw_smth(event):
    global lasx, lasy, color
    brushSize = brushSlider.get()
    drawMask.ellipse([lasx - brushSize, lasy - brushSize, lasx + brushSize, lasy + brushSize], fill=color)
    lasx, lasy = event.x, event.y

    updateImgSlot(2)

def img2imgFunc():
    global pipeline, pipelinePaint, image2
    if "pipeline" in globals():
        del (pipeline)
    if "pipelinePaint" in globals():
        del (pipelinePaint)
    if "pipelineImg2Img" not in globals():
        loadImg2ImgSD()

    strenght = float(StrengthSlider.get()) / 10

    with autocast(device):
        imageOut = pipelineImg2Img(prompt=prompt.get() + "," + artstylesList[int(v.get()) - 1] + ",artstyle",
                            num_inference_steps=stepSlider.get(),
                            negative_prompt=promptN.get(),
                            guidance_scale=guidanceScaleSlider.get(),
                            init_image=canvas,
                            strength=strenght,
                            generator=generator).images[0]

        image2 = imageOut
        updateImgSlot(2)





def generateNewImage():
    global pipelinePaint, image1, pipelineImg2Img
    if "pipelinePaint" in globals():
        del (pipelinePaint)
    if "pipelineImg2Img" in globals():
        del (pipelineImg2Img)
    if "pipeline" not in globals():
        loadComicSD()

    with autocast(device):
        imageOut = pipeline(prompt=prompt.get() + "," + artstylesList[int(v.get()) - 1] + ",artstyle",
                            num_inference_steps=stepSlider.get(),
                            negative_prompt=promptN.get(),
                            guidance_scale=guidanceScaleSlider.get(),
                            generator=generator).images[0]
    image1 = imageOut
    updateImgSlot(1)


def editCurrentImage():
    global pipeline, image3, pipelineImg2Img
    if "pipeline" in globals():
        del (pipeline)
    if "pipelineImg2Img" in globals():
        del (pipelineImg2Img)
    if "pipelinePaint" not in globals():
        loadPaintingSD()

    mask = PIL.ImageOps.invert(imageMask)
    mask.save("massska.png")
    with autocast(device):
        imageOut = pipelinePaint(prompt=prompt.get() + "," + artstylesList[int(v.get()) - 1] + ",artstyle",
                                 image=image2, mask_image=mask, guidance_scale=guidanceScaleSlider.get(),
                                 num_inference_steps=stepSlider.get(), generator=generator,
                                 negative_prompt=promptN.get()).images[0]



    image3 = imageOut
    imageOut.save(f"generatko.png")
    updateImgSlot(3)


def updateImgSlot(slot):
    global image1, image2, image3, canvasLabel, canvas, blobMask
    if slot == 1:
        imgToDisplay = ImageTk.PhotoImage(image1)
        imgSlot1.configure(image=imgToDisplay)
        imgSlot1.image = imgToDisplay

    elif slot == 2:
        # Create image from mask and current image
        imageAndMask = Image.composite(image2, whiteImg, imageMask)
        imgToDisplay = ImageTk.PhotoImage(imageAndMask)
        imgSlot2.configure(image=imgToDisplay)
        imgSlot2.image = imgToDisplay

    elif slot == 3:
        # Create image from mask and current image
        imageAndMask = Image.composite(image3, messageImg, blobMask)
        imgToDisplay = ImageTk.PhotoImage(imageAndMask)
        imgSlot3.configure(image=imgToDisplay)
        imgSlot3.image = imgToDisplay

    elif slot == 4:
        imgToDisplay = ImageTk.PhotoImage(canvas)
        canvasLabel.configure(image=imgToDisplay)
        canvasLabel.image = imgToDisplay

# Mask functions:

def redoMask():
    global imageMask, drawMask
    imageMask = Image.new("L", (512, 512), 255)
    drawMask = ImageDraw.Draw(imageMask)
    updateImgSlot(2)


def clearCanvas():
    global canvas, canvasDraw
    canvas = Image.new("RGB", (512, 512))
    canvasDraw = ImageDraw.Draw(canvas)
    updateImgSlot(4)


def invertMask():
    global imageMask, INVERT, color, whiteImg, drawMask
    imageMask = PIL.ImageOps.invert(imageMask)
    # drawMask = imageMask
    updateImgSlot(2)
    if INVERT == 0:
        INVERT = 1
        color = 'white'
    else:
        INVERT = 0
        color = 'black'
    print(color)


# Button functions:

def useThisImage():
    global image2
    image2 = image1.copy()
    updateImgSlot(2)
    print("a")


def useThisImage2():
    global image3
    image3 = image2.copy()
    updateImgSlot(3)


def useThisImage3():
    global image2
    image2 = image3.copy()
    updateImgSlot(2)


# Dialog box:


def deleteAllDialog():
    global messageImg, blobMask
    blobMask = Image.new("L", (512, 512), 255)
    messageImg = Image.new("RGB", (512, 512), (255, 255, 255))
    updateImgSlot(3)

def placeDialogBox(event):
    global messages, currentmessages, messageImg, blobMask
    x, y = event.x, event.y
    messageToAdd = dialogText.get("1.0", END)
    messageToAdd = messageToAdd.upper()
    fontSize = fontSlider.get()
    font = ImageFont.truetype('comic.ttf', fontSize)

    if messages == 1:

        messageImgDraw = ImageDraw.Draw(messageImg)
        blobMaskDraw = ImageDraw.Draw(blobMask)

        text_width, text_height = messageImgDraw.textsize(messageToAdd, font=font)

        messageImgDraw.rectangle((x - margain, y - margain, x+text_width + margain, y+text_height - 15),
                            fill="white", outline="black")
        blobMaskDraw.rectangle((x - margain, y - margain, x+text_width + margain, y+text_height - 15),
                            fill="black", outline="black")
        messageImgDraw.text((x, y), messageToAdd, fill="black", font=font)
        messages = 0
        updateImgSlot(3)

    else:
        messageImg = Image.new("RGB", (512, 512))
        blobMask = Image.new("L", (512, 512), 255)
        messages = 1
        placeDialogBox(event)



#####################################################################################################################

def randomize():
    global generator
    random_int = random.randint(1, 2147483647)
    generator = torch.Generator("cuda").manual_seed(random_int)


def pickedMe(index):
    # bindas pred tem vse label in potem lamba samo indeks nad katero smo stisnili
    global image2
    print(index)
    image2 = imageList[index].copy()
    updateImgSlot(2)


def updateMyLabels(index):
    # tisto, ki smo dodali sliko updajettj user-ju
    img = imageList[index].resize((128, 128))
    imgToDisplay = ImageTk.PhotoImage(img)
    labele[index].configure(image=imgToDisplay)
    labele[index].image = imgToDisplay


def saveToList():
    global image2, pointer, imageList
    if pointer > 9:
        pointer = 0
    imageToAdd = image2.copy()
    imageList.insert(pointer, imageToAdd)
    updateMyLabels(pointer)
    pointer += 1

def addToFinalImage():
    global image3, stevecX, stevecY

    # if stevecX >= 3:
    #     stevecX = 0
    #     stevecY += 1

    final_image.paste(image3, (coordsX[stevecX], coordsY[stevecY]))
    # stevecX += 1
    # Update to user:
    imgToDisplay = final_image.resize((414, 547))
    imgToDisplay = ImageTk.PhotoImage(imgToDisplay)
    finalLabel.configure(image=imgToDisplay)
    finalLabel.image = imgToDisplay

def exportComic():
    final_image_under.paste(final_image, (40, 40))
    final_image_under.save(f"GeneratedComic.pdf")

def saveImages():
    global imageList

    for i in range(0, len(imageList)):
        imageList[i].save(directory + "\SavedImg" + str(i) + ".png")

# APP:
app = Tk()
app.geometry("2500x1400")
app.title("Comic Generator")

# Variables:
INVERT = 0
color = 'black'
device = "cuda"
messages = 1
currentmessages = 0
margain = 5
pointer = 0
imageList = []
coordsX = [0, 512+20, 512+512+40]
coordsY = [0, 512+20, 512+521+20+10, 512+512+512+60]
stevecX = 0
stevecY = 0

directory = f".\\SavedImages"

# Final Image:
final_image = Image.new("RGB", (1576, 2108))
final_image_under = Image.new("RGB", (1656, 2188), (255, 255, 255))

# Generator:
generator = torch.Generator("cuda").manual_seed(2229135949491605)

# Box:
boxCoordinates = []

# Images:
image1 = Image.open("astronaut_rides_horse4.png")
image2 = Image.new("RGB", (512, 512))
image3 = Image.new("RGB", (512, 512))

imageMask = Image.new("L", (512, 512), 255)  # white mask
blobMask = Image.new("L", (512, 512), 255)
messageImg = Image.new("RGB", (512, 512), (255, 255, 255)) # tuki gor gre dialog boxi
whiteImg = Image.new("RGB", (512, 512), (255, 255, 255))  # used for displaying to user
drawMask = ImageDraw.Draw(imageMask)  # mask to be edited by user
demo = ImageTk.PhotoImage(image1)  # display demo startup image

# Image widgets:

imgSlot1 = Label(app, image=demo)
imgSlot1.place(x=150+200, y=160)

imgSlot2 = Label(app, image=demo)
imgSlot2.place(x=512 + 150 + 200 + 44, y=160)

imgSlot3 = Label(app, image=demo)
imgSlot3.place(x=512 * 2 + 44 * 2 + 150 + 200, y=160)

# Binding functions to images:

imgSlot2.bind("<Button-1>", get_x_and_y)
imgSlot2.bind("<B1-Motion>", draw_smth)

# imgSlot3.bind("<Button-1>", getFirstCoordinates)
# imgSlot3.bind("<ButtonRelease-1>", getLastCoordinates)

imgSlot3.bind("<Button-1>", placeDialogBox)

# Canvas:

# Define the set_color function

paintColor = "black"

def set_color(color):
    global paintColor
    paintColor = color


def paint(event):
    x, y = event.x, event.y
    size = paintBrushSlider.get()
    canvasDraw.ellipse([x-size, y-size, x+size, y+size], fill=paintColor)
    updateImgSlot(4)

# Create a canvas
canvas = Image.new("RGB", (512, 512))
canvasDraw = ImageDraw.Draw(canvas)
canvasToDisplay = ImageTk.PhotoImage(canvas)
canvasLabel = Label(app, image=canvasToDisplay)
# Create widgete
canvasLabel.place(x=150+200, y=800)

# Bind the "Motion" event to the paint function
canvasLabel.bind("<B1-Motion>", paint)


# Canvas options:

canvasOptions = tkinter.Label(app, font=("Rockwell bold", 16), text="Canvas options:")
canvasOptions.place(x=10, y=800)

brus = tkinter.Label(app, font=("Rockwell", 14), text="Painting brush size:")
brus.place(x=10, y=850)

colors = tkinter.Label(app, font=("Rockwell", 14), text="Select color:")
colors.place(x=10, y=1010)

strenght = tkinter.Label(app, font=("Rockwell", 14), text="Strength:")
strenght.place(x=10, y=930)

StrengthSlider = Scale(app, from_=1, to=10, length=300, orient=HORIZONTAL)
StrengthSlider.place(x=10, y=960)
StrengthSlider.set(8)

paintBrushSlider = Scale(app, from_=1, to=40, length=300, orient=HORIZONTAL)
paintBrushSlider.place(x=10, y=880)
paintBrushSlider.set(25)

red_button = Button(app, text="", bg="red", command=lambda: set_color("red"), width=2)
red_button.place(x=10, y=1050)

green_button = Button(app, text="", bg="green",  command=lambda: set_color("green"),width=2)
green_button.place(x=10 + 30, y = 1050)

blue_button = Button(app, text="", bg="blue", command=lambda: set_color("blue"),width=2)
blue_button.place(x=10 + 2*30, y=1050)

gray_button = Button(app, text="", bg="gray", command=lambda: set_color("gray"),width=2)
gray_button.place(x=10 + 3*30, y=1050)

black_button = Button(app, text="", bg="black", command=lambda: set_color("black"),width=2)
black_button.place(x=10 + 4*30, y=1050)

white_button = Button(app, text="", bg="white", command=lambda: set_color("white"),width=2)
white_button.place(x=10 + 5*30, y=1050)

brown_button = Button(app, text="", bg="brown", command=lambda: set_color("brown"),width=2)
brown_button.place(x=10 + 6*30, y=1050)

orange_button = Button(app, text="", bg="orange", command=lambda: set_color("orange"),width=2)
orange_button.place(x=10 + 7*30, y=1050)

yellow_button = Button(app, text="", bg="yellow", command=lambda: set_color("yellow"),width=2)
yellow_button.place(x=10 + 8*30, y=1050)

clearCanvasButton = ctk.CTkButton(font=("Rockwell bold", 14), height=40, width=80, text_color="white", fg_color="blue",
                           command=clearCanvas, master=app)
clearCanvasButton.configure(text="Clear")
clearCanvasButton.place(x=120, y=1150)



# WIDGETS - UI elements:

promptText = tkinter.Label(app, font=("Rockwell bold", 18), text="Positive Prompt:")
promptText.place(x=10, y=10)

NpromptText = tkinter.Label(app, font=("Rockwell bold", 18), text="Negative Prompt:")
NpromptText.place(x=10, y=60)

pickArtstyle = tkinter.Label(app, font=("Rockwell bold", 16), text="Select Art-style:")
pickArtstyle.place(x=10, y=120)


prompt = ctk.CTkEntry(height=40, width=800, text_color="black", fg_color="white", master=app)
prompt.place(x=150+80, y=10)

promptN = ctk.CTkEntry(height=40, width=800, text_color="black", fg_color="white", master=app)
promptN.place(x=150+80, y=60)

generateImgButton = ctk.CTkButton(height=40, width=120, text_color="white", fg_color="blue", command=generateNewImage,
                                  master=app)
generateImgButton.configure(text="Generate new image", font=("Rockwell bold", 14))
generateImgButton.place(x=540-10, y=700-15)

editImgButton = ctk.CTkButton(height=40, width=120, text_color="white", fg_color="blue", command=editCurrentImage,
                              font=("Rockwell bold", 14), master=app)
editImgButton.configure(text="Edit current image")
editImgButton.place(x=530 + 512+44+20+45-7, y=700-15)

editImgButton = ctk.CTkButton(height=40, width=120, text_color="white", fg_color="blue", command=saveToList,
                              font=("Rockwell bold", 14), master=app)
editImgButton.configure(text="Save for later")
editImgButton.place(x=530 + 300+85+5, y=700-15)


spust = 60


def loadImage():
    global image2
    newPop = Tk()
    newPop.withdraw()
    file_path = filedialog.askopenfilename()
    loadedImage = Image.open(file_path)
    image2 = loadedImage.copy()
    updateImgSlot(2)


redoMaskButton = ctk.CTkButton(height=40, width=80, text_color="white", fg_color="blue", command=redoMask, master=app,
                               font=("Rockwell bold", 14))
redoMaskButton.configure(text="Redo")
redoMaskButton.place(x=512 + 150 + 200 + 44, y=720+spust)


invertMaskButton = ctk.CTkButton(height=40, width=80, text_color="white", fg_color="blue", command=invertMask,
                                 font=("Rockwell bold", 14), master=app)
invertMaskButton.configure(text="Invert")
invertMaskButton.place(x=512 + 150 + 200 + 44 + 90, y=720+spust)

loadButton = ctk.CTkButton(font=("Rockwell bold", 14), height=40, width=80, text_color="white", fg_color="blue",
                           command=loadImage, master=app)
loadButton.configure(text="Load")
loadButton.place(x=530 + 512+44+20-45-7, y=700-15)

clearPrompt = ctk.CTkButton(height=30, width=20, text_color="white", fg_color="red", master=app,
                            font=("Rockwell bold", 19), command=lambda: prompt.delete(0, END))
clearPrompt.configure(text="X")
clearPrompt.place(x=955+80, y=15)

clearPromptN = ctk.CTkButton(height=30, width=20, text_color="white", fg_color="red", master=app,
                             font=("Rockwell bold", 19), command=lambda: promptN.delete(0, END))
clearPromptN.configure(text="X")
clearPromptN.place(x=955+80, y=15+50)

img2imgButton = ctk.CTkButton(height=30, width=20, text_color="white", fg_color="blue", master=app,
                             font=("Rockwell bold", 16), text="Generate image from my image", command=img2imgFunc)
img2imgButton.place(x=30, y=1200)



useThisImageButton = ctk.CTkButton(height=30, width=15, text_color="white", fg_color="blue", master=app,
                                   command=useThisImage, font=("Rockwell bold", 25))
useThisImageButton.configure(text=">")
useThisImageButton.place(x=872, y=400)






### DIALOG BOX:


generation = tkinter.Label(app, font=("Rockwell bold", 16), text="Insert text:")
generation.place(x=2000, y=160)

dialogText = Text(master=app, width=42, height=15, font=("Ariel", 14))
dialogText.place(x=2000, y=190)


a = tkinter.Label(app, font=("Rockwell bold", 16), text="Font size:")
a.place(x=2000, y=530)

fontSlider = Scale(app, from_=10, to=28, length=300, orient=HORIZONTAL)
fontSlider.place(x=2000, y=570)
fontSlider.set(22)

def addMessage():
    global messages, image3, messageImg, blobMask
    messages = 1
    # tu zapeci prejsno masko z image3
    invertko = PIL.ImageOps.invert(blobMask)
    image4 = image3.paste(messageImg, (0, 0), invertko)
    image3 = image4.copy()

placeDialogButton = ctk.CTkButton(height=40, width=80, text_color="white", fg_color="blue", master=app,font=("Rockwell bold", 16),
                                  command=addMessage)
placeDialogButton.configure(text="Confirm placement")
placeDialogButton.place(x=2000, y=635)

saveImgButton = ctk.CTkButton(height=40, width=80, text_color="white", fg_color="blue", master=app,font=("Rockwell bold", 16),
                                  command=addToFinalImage)
saveImgButton.configure(text="Add to comic")
saveImgButton.place(x=2160 + 20, y=635)

def newSlot():
    global stevecY, stevecX
    if stevecX > 1:
        stevecX = 0
        stevecY += 1
        stevecX -= 1

    stevecX += 1



def setSlot(a):
    global stevecY, stevecX
    b = int(a[5:])

    if b == 1:
        stevecX = 0
        stevecY = 0
    elif b == 2:
        stevecX = 1
        stevecY = 0
    elif b == 3:
        stevecX = 2
        stevecY = 0
    elif b == 4:
        stevecX = 0
        stevecY = 1
    elif b == 5:
        stevecX = 1
        stevecY = 1
    elif b == 6:
        stevecX = 2
        stevecY = 1
    elif b == 7:
        stevecX = 0
        stevecY = 2
    elif b == 8:
        stevecX = 1
        stevecY = 2
    elif b == 9:
        stevecX = 2
        stevecY = 2
    elif b == 10:
        stevecX = 0
        stevecY = 3
    elif b == 11:
        stevecX = 1
        stevecY = 3
    elif b == 12:
        stevecX = 2
        stevecY = 3



newSlotButton = ctk.CTkButton(height=40, width=80, text_color="white", fg_color="blue", master=app,font=("Rockwell bold", 16),
                                  command=lambda: setSlot(combo.get()))
newSlotButton.configure(text="New slot")
newSlotButton.place(x=2160, y=635+50)

combo = tkinter.ttk.Combobox(app, values=["Slot 1", "Slot 2", "Slot 3", "Slot 4",
                                   "Slot 5", "Slot 6", "Slot 7", "Slot 8",
                                   "Slot 9", "Slot 10", "Slot 11", "Slot 12"])
combo.place(x=2160+100, y=635+50)





removeDialogButton = ctk.CTkButton(height=40, width=80, text_color="white", fg_color="blue", master=app,font=("Rockwell bold", 16),
                                  command=deleteAllDialog)
removeDialogButton.configure(text="Remove dialog")
removeDialogButton.place(x=2160 + 150, y=635)

saveImagesButton = ctk.CTkButton(height=40, width=80, text_color="white", fg_color="blue", master=app,font=("Rockwell bold", 16),
                                  command=saveImages)
saveImagesButton.configure(text="Save images")
saveImagesButton.place(x=2160 + 150, y=635+100)


################





useThisImageButton2 = ctk.CTkButton(height=30, width=15, text_color="white", fg_color="blue", master=app,
                                    font=("Rockwell bold", 25), command=useThisImage2)
useThisImageButton2.configure(text=">")
useThisImageButton2.place(x=900+527, y=400+20)

useThisImageButton3 = ctk.CTkButton(height=30, width=15, text_color="white", fg_color="blue", master=app,
                                    font=("Rockwell bold", 25), command=useThisImage3)
useThisImageButton3.configure(text="<")
useThisImageButton3.place(x=900+527, y=400-20)

v = StringVar(app, "1")

artstyles = {"Andreasrocha": "1",
             "Charliebo": "2",
             "Holliemengert": "3",
             "Marioalberti": "4",
             "Pepelarraz": "5",
             "Jamesdaly": "6"}

artstylesList = ["Andreasrocha", "Charliebo", "Holliemengert", "Marioalberti", "Pepelarraz", "Jamesdaly"]

currentY = 0
for (text, artstyles) in artstyles.items():
    if int(artstyles) % 2 == 1:
        Radiobutton(app, text=text, variable=v, value=artstyles, font=("Rockwell", 14)).place(x=10, y=160 + 20 * (int(artstyles) - 1))
        currentY = 160 + 20 * (int(artstyles) - 1)
    else:
        Radiobutton(app, text=text, variable=v, value=artstyles, font=("Rockwell", 14)).place(x=10 + 180, y=currentY)




# General sliders:

options = tkinter.Label(app, font=("Rockwell bold", 16), text="General options:")
options.place(x=10, y=295)

odstevek = 15

steps = tkinter.Label(app, font=("Rockwell", 14), text="Number of steps:")
steps.place(x=10, y=350 - odstevek)

stepSlider = Scale(app, from_=10, to=110, length=300, orient=HORIZONTAL)
stepSlider.place(x=10, y=385-10 - odstevek)
stepSlider.set(50)





Guidance = tkinter.Label(app, font=("Rockwell", 14), text="Guidance scale:")
Guidance.place(x=10, y=350+100 - odstevek*2)

guidanceScaleSlider = Scale(app, from_=1, to=30, length=300, orient=HORIZONTAL)
guidanceScaleSlider.place(x=10, y=385+100-10 - odstevek*2)
guidanceScaleSlider.set(8)




numOfImg = tkinter.Label(app, font=("Rockwell", 14), text="Number of images:")
numOfImg.place(x=10, y=350+100+100-odstevek*3)

ifgenerator = tkinter.Label(app, font=("Rockwell", 14), text="Generator:")
ifgenerator.place(x=10, y=350+100+100+100-odstevek*4)

textSeed = ctk.CTkButton(app, font=("Rockwell bold", 16), text=" Randomize seed", command=randomize, text_color="white", fg_color="blue")
textSeed.place(x=10, y=350+100+100+150-odstevek*5)

numberOfImages = Scale(app, from_=1, to=5, length=300, orient=HORIZONTAL)
numberOfImages.place(x=10, y=385+200-10-odstevek*3)
numberOfImages.set(3)


# Mask:

maskname = tkinter.Label(app, font=("Rockwell bold", 16), text="Mask options:")
maskname.place(x=512 + 150 + 200 + 44, y=160+512+10+spust)

maskname = tkinter.Label(app, font=("Rockwell bold", 16), text="Mask brush size:")
maskname.place(x=512 + 150 + 200 + 44 + 250 + 30, y=160+512+10+spust)


brushSlider = Scale(app, from_=1, to=40, length=300, orient=HORIZONTAL)
brushSlider.place(x=512 + 150 + 200 + 44 + 90 + 120, y=720+spust-5)
brushSlider.set(25)


# Splosne zadeve:

generation = tkinter.Label(app, font=("Rockwell bold", 16), text="GENERATION:")
generation.place(x=530, y=120)

img2img = tkinter.Label(app, font=("Rockwell bold", 16), text="IMG-2-IMG:")
img2img.place(x=540, y=760)

editing = tkinter.Label(app, font=("Rockwell bold", 16), text="EDITING:")
editing.place(x=530 + 512+44+20, y=120)

results = tkinter.Label(app, font=("Rockwell bold", 16), text="RESULT:")
results.place(x=530+512*2+44*2+20+20, y=120)


pick = tkinter.Label(app, font=("Rockwell bold", 16), text="Pick image to edit:")
pick.place(x=512 + 150 + 200 + 44, y=850)

testImg = Image.new("RGB", (128, 128))
testImgDisplay = ImageTk.PhotoImage(testImg)




labels = ['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label 6', 'Label 7', 'Label 8', 'Label 9', 'Label 10']

# Create a label for each string in the list
labele = []

# Create a list of lambda functions, each with its own index argument
lambdas = [lambda event, index=i: pickedMe(index) for i in range(len(labels))]

for i, label in enumerate(labels):
    if i < 5:
        lab = tk.Label(app, image=testImgDisplay)
        lab.place(x=906 + i * 140, y=880)
        # Bind the lambda function for this label to the label's '<B1-Motion>' event
        lab.bind("<Button-1>", lambdas[i])
        labele.append(lab)
    else:
        lab = tk.Label(app, image=testImgDisplay)
        lab.place(x=906 + (i-5) * 140, y=880 + 128 + 20)
        # Bind the lambda function for this label to the label's '<B1-Motion>' event
        lab.bind("<Button-1>", lambdas[i])
        labele.append(lab)


finalLabel = Label(app, image=demo)
finalLabel.place(x=1700, y=800)



exportComicButton = ctk.CTkButton(app, font=("Rockwell bold", 16), text=" Export comic", command=exportComic, text_color="white", fg_color="blue")
exportComicButton.place(x=2150, y=1300)





app.mainloop()


