I want create a dataset for 
Image Forgery Detection and Localization task


drawthingsai/megalith-10m


Collect Real Images:
Use megalith-10m as our source 


Forgery Image:
Traditional Operations:
Copy-Move
Image Splicing
Removal

AIGC Operations:
- Pure Generation
- Inpainting / Region Edition / Replace
- Outpainting
- Erase
- Change Background
- Add New Object 




Log:
00013 Flux inpaint

000065 SD15 inpaint



























for each operation we need write code to implement
but we have to design the pipline for each operation first


We have to select the AIGC models, 

Pure Generation Pipeline: 
a. get the prompt
b. Use AI model to generate an image
c. Save the generated image

Inpainting / Region Edition / Replace Pipeline:
a. Load a real image
b. Generate or select a mask for the region to be edited
c. Generate a prompt for the new content
d. Use AI model to inpaint the masked region
e. Save the edited image

Change Background Pipeline:
a. Load a real image
b. Segment the foreground object
c. Generate a new background prompt
d. Use AI model to generate a new background
e. Compose the foreground object with the new background
f. Save the edited image

Add New Object Pipeline:
a. Load a real image
b. Generate a prompt for a new object
c. Use AI model to generate the new object
d. Blend the new object into the original image
e. Save the edited image


Outpainting Pipeline:
a. Load a real image
b. Expand the canvas size
c. Generate a prompt for the new content
d. Use AI model to outpaint the expanded regions
e. Save the outpainted image

Erase Pipeline:
a. Load a real image
b. Select an object or region to erase
c. Create a mask for the selected region
d. Use AI model to fill the erased region
e. Save the edited image











try sdxl:

I want to do automatically Inpainting / Region Edition / Replace pipeline of a given image
to create a dataset of image manimuplation detection.
So i need LLM to give instruction of which part to replace and the target content
here are steps i plain

1st, use florence to caption (or just use provided caption)
2nd, ask LLM to choose an which region to edit (inpaint), and get the target content of the region want to replace
3rd, use SAM 2 to ground segmentation this part
4th, use mask+image+new caption to inpaint

give me the prompts i need to give llm to get what i want

(SD1.5 SD2 SDXL SD3 FLUX)

















