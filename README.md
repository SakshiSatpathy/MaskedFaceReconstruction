# MaskedFaceReconstruction
### Pipeline:
1. `sam_experiment.py`:<br>
This code allows the user to segment the original images using SAM.<br>
Note: you have to manually update the filepaths within the code.<br> You also have to download the [ViT-H SAM Model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file) and put it under the same directory of the file.
Running the code will prompt a small GUI to select points to include/exclude in the segmented image.<br>
Left-clicking on a point (shown as a blue dot) will INCLUDE that point, and Right-Clicking (shown as a red dot) will EXCLUDE that point.
<br><br>
2. [3DMM-Fitting-Pytorch](https://github.com/ascust/3DMM-Fitting-Pytorch/tree/master): <br><br>
To run the code, you must download the [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads) and place `01_MorphableModel.mat` into the folder `3DMM-Fitting-Pytorch/BFM`<br><br>
You must also download the Expression Basis. Go to the [repo](https://github.com/Juyong/3DFace), download the "CoarseData" and put "Exp_Pca.bin" into "BFM".<br><br>
After that, you must convert the BFM parameters by running `python convert_bfm09_data.py` under "3DMM-Fitting-Pytorch".<br><br>
To run the model, run `python fit_single_img.py --img_path [filepath] --res_folder results`. This will output the .obj file to results, and will output 5 different angles of the rendered face into the current working directory, set as `angled_image_{0-5}.jpg`. <br><br>
I modified the file `3DMM-Fitting-Pytorch/core/BFM09Model.py` to output the 5 angles of rendered images. If you would like to modify the angles, modify the list `ROTATION_ANGLES` on line 117.
<br><br>
3. Preprocessing before GAN <br><br>
Modify `generate_binary_mask.py` with the filename of the desired angled image (from the 3DMM), and run it. This file is used for preprocessing before running images through EdgeConnect. When run, the code prompts the user to draw on an image, creating a white binary mask.
As required by EdgeConnect, this generates a white binary mask, an image with the binary mask overlayed, and a Canny edge-map to guide EdgeConnect.
This binary mask represents the area that must be infilled by EdgeConnect. <br><br>
To try fusing the edges before feeding into EdgeConnect, run `fusing_edges.py`. This averages all of the edge-maps. In my testing, there weren't any significant differences from just using the frontal edge.
<br><br>
To preprocess for LAMA instead, run `generate_binary_mask_lama.py`. This creates a black binary mask instead, as required by LAMA, and doesn't include an edge-map.
<br><br>
4. Running [EdgeConnect](https://github.com/knazeri/edge-connect)<br><br>
To run EdgeConnect, move the files of the binary mask, the image with the binary mask overlayed, and the edge map into the folder `edge-connect`.<br><br>
Before running, download the [CelebA Models](https://drive.google.com/drive/folders/13JgMA5sKMYgRwHBp4f7PBc5orNJ_Cv-p) and place it under the `edge-connect/checkpoints` folder. <br><br>
Then run: 
`python test.py --model 2 --checkpoints ./checkpoints --input [path to file with input image with binary mask overlayed]  --mask [path to binary mask file]  --output [path to the output directory]  --edge [path to edge map]`
<br><br>
5. Running [LaMa](https://github.com/advimman/lama) (instead of EdgeConnect)<br><br>
To run LAMA instead, navigate to the directory `lama` and run: `export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)`.<br><br>
Then go to this [link](https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips?usp=drive_link) and download `LaMa_models.zip` and unzip the folder under `lama`.<br><br>
Move all input images and input masks to the `LaMa_test_images` directory. Name the inputs to follow the format: `image1_mask001.png` for the binary mask file, and `image1.png` for the image. <br><br> Then run:
`python bin/predict.py model.path=$(pwd)/LaMa_models/lama-celeba-hq/lama-regular indir=$(pwd)/LaMa_test_images outdir=$(pwd)results`<br><br>
In my testing, the LaMa model was significantly worse in performance when compared to EdgeConnect. LaMa seems to be oriented to architectural infilling, rather than facial infilling.

### Outputs
All outputs are in the folder `experiment_outputs`. Each subfolder `person_{i}` contains: <br><br><br>
`mask_test_{i}.jpg/png`: the original image<br>
`mask_test_{i}_segmented.jpg`: the SAM output <br>
`mask_test_{i}_segmented_mesh.obj`: the .obj output of the 3DMM <br>
`mask_test_{i}_segmented_composed_image.jpg`: the image output of the rendered 3DMM mesh overlayed on top of segmented image <br>
`angled_img_{0-4}.jpg`: the rendered image of the 3DMM mesh from 5 different angles, including the frontal view. <br>
`attempt_1/binary_mask.jpg`: the binary mask to be infilled by EdgeConnect (output of `generate_binary_mask.py`)<br>
`attempt_1/edge_map.jpg`: the Canny edge map to guide EdgeConnect infilling (output of `generate_binary_mask.py`)<br>
`attempt_1/img_and_bm.jpg`: the rendered frontal image of 3DMM mesh with binary mask overlayed for EdgeConnect (output of `generate_binary_mask.py`)<br>
`attempt_1/output.jpg`: the EdgeConnect output

<br><br> For the first person, I ran several attempts. I first infilled his face from a side view, and then tried a frontal view. On my third attempt, I tried using fused edges with EdgeConnect, but it made no significant improvement in quality. Finally, `experiment_outputs/person_1/lama_attempt` shows the output of the LaMa trial, which did extremely poorly. After the first person, I focused on just doing frontal views using EdgeConnect. 

<br><br> There is also a folder `experiment_outputs/unmasked_3dmm_output`, which shows how the 3DMM performs with unmasked face images.

### Notes
`generate_views.py` is not used at all. It served as the original approach taken to load textures from .obj output of 3DMM.
Originally, I modified the 3DMM-Fitting-Pytorch code to output a textured image. These attempts would fail due to incorrectly named variables.
Additionally, Pytorch3D requires .mtl files to load the textured mesh, which I was unable to find outputted in the 3DMM-Fitting-Pytorch.
This code was made to load the outputted mesh through Open3D, and load the textured image on top of it, and then convert it to Pytorch3D.
This conversion would also lead to several errors. In the end, I finally found the correctly named rendered image from 3DMM-Fitting-Pytorch, and outputted it directly, effectively deprecating this file.
