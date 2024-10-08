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
Flux
00013 Flux inpaint

SD15
00065 SD15 inpaint
00048 SD15 inpaint
00370 SD15 inpaint
00099 SD15 inpaint
00115 SD15 inpaint
00073 SD15 inpaint
00015 SD15 inpaint


SD3 CN
00195 SD3 inpaint




SD2









SDXL (need Fooocus)
Kolors
SC CN (official)
pixart_inpaint.py

PowerPaint
BrushNet
Paint by Example



























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





























