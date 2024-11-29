
drawthingsai/megalith-10m

Collect Real Images:
Use megalith-10m as our source

Forgery Image:

Traditional Operations:
- Copy-Move
- Image Splicing
- Removal

AIGC Operations:
- Pure Generation
- Inpainting / Region Edition / Replace
- Erase
- Change Background
- Add New Object 
- Outpainting


Log:
wget https://huggingface.co/datasets/drawthingsai/megalith-10m/resolve/main/00197.tar
mkdir -p 00197/raw_images && tar -xvf 00197.tar -C 00197/raw_images

Flux.1 dev
00013 Flux inpaint
00016 Flux inpaint
00017 Flux inpaint
00018 Flux inpaint
00019 Flux inpaint
00020 Flux inpaint
00021 xxxx dont care somethingwrong
00022 Flux inpaint
00023 Flux inpaint
00024 Flux inpaint
00025 Flux inpaint
00026 Flux inpaint
00027 Flux inpaint
00028 Flux inpaint

SD15
00015 SD15 inpaint
00048 SD15 inpaint
00065 SD15 inpaint
00073 SD15 inpaint
00099 SD15 inpaint
00370 SD15 inpaint
00115 SD15 inpaint
00371 SD15 inpaint
00372 SD15 inpaint
00373 SD15 inpaint
00014 SD15 inpaint

SD2
00200 SD2 inpaint
00201 SD2 inpaint
00202 SD2 inpaint
00203 SD2 inpaint
00204 SD2 inpaint
00205 SD2 inpaint
00206 SD2 inpaint
00207 SD2 inpaint
00208 SD2 inpaint
00209 SD2 inpaint

SD3
00188 SD3 inpaint
00189 SD3 inpaint
00190 SD3 inpaint
00191 SD3 inpaint
00192 SD3 inpaint
00193 SD3 inpaint
00194 SD3 inpaint
00195 SD3 inpaint
00196 SD3 inpaint
00197 SD3 inpaint
00198 SD3 inpaint
00199 SD3 inpaint

SDXL diffusers/stable-diffusion-xl-1.0-inpainting-0.1
00210 SDXL inpaint 
00211 SDXL inpaint 
00212 SDXL inpaint 
00213 SDXL inpaint 
00214 SDXL inpaint 
00215 SDXL inpaint 
00216 SDXL inpaint 
00217 SDXL inpaint 
00218 SDXL inpaint 
00219 SDXL inpaint 
00220 SDXL inpaint 

SD3.5L
00300
00301
00302
00303
00304
00305
00306

Flux.1 Fill dev
00320 d
00321
00322
00323
00324
00325
00326







wget https://huggingface.co/datasets/drawthingsai/megalith-10m/resolve/main/00188.tar
mkdir -p 00188/raw_images && tar -xvf 00188.tar -C 00188/raw_images

ideogram_remix_V_2 Dreamapi
PowerPaint v1    5k
PowerPaint v2    5k
BrushNet         5k
Kolors (official) 5k
pixart_inpaint    5k
Kandinsky 3.1  https://github.com/ai-forever/kandinsky-3
HunyuanDiT 
Kolors Virtual Try-On in the Wild
SANA
levihsu/OOTDiffusion



_Full Gen:
SD15
SD21
SDXL
SD3
Flux.1
Playground2.5
PixArt alpha
PixArt sigma
unidiffuser
Stable Cascade




















