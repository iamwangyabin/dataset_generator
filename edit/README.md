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




https://huggingface.co/datasets/drawthingsai/megalith-10m/resolve/main/00371.tar
mkdir -p 00371/raw_images && tar -xvf 00371.tar -C 00371/raw_images



SD15
00015 SD15 inpaint
00048 SD15 inpaint
00065 SD15 inpaint
00073 SD15 inpaint
00099 SD15 inpaint
00115 SD15 inpaint
00370 SD15 inpaint
00371 SD15 inpaint
00372
00373

SD2
00200 SD2 inpaint

SD3
00195 SD3 inpaint


SDXL (original)
00210




SDXL (Fooocus)







SC CN (official)






Kolors (official)
 







pixart_inpaint.py
PowerPaint
BrushNet
Paint by Example



















































