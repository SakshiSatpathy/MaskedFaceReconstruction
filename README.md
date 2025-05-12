# MaskedFaceReconstruction
### Pipeline:
1. `sam_experiment.py`:<br>
This code allows the user to segment the original images using SAM.<br>
Note: you have to manually update the filepaths within the code.<br>
Running the code will prompt a small GUI to select points to include/exclude in the segmented image.<br>
Left-clicking on a point (shown as a blue dot) will INCLUDE that point, and Right-Clicking (shown as a red dot) will EXCLUDE that point.
<br><br>
2. [3DMM-Fitting-Pytorch](https://github.com/ascust/3DMM-Fitting-Pytorch/tree/master): <br><br>
To run the code, you must download the [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads) and place `01_MorphableModel.mat` into the folder `3DMM-Fitting-Pytorch/BFM`<br><br>
You must also download the Expression Basis. Go to the [repo](https://github.com/Juyong/3DFace), download the "CoarseData" and put "Exp_Pca.bin" into "BFM".<br><br>
After that, you must convert the BFM parameters by running `python convert_bfm09_data.py` under "3DMM-Fitting-Pytorch".<br><br>
To run the model, run `python fit_single_img.py --img_path [filepath] --res_folder results`. This will output the .obj file to results, and will output 5 different angles of the rendered face into the current working directory, set as "angled_image_{0-5}.jpg". <br><br>
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

4. Running EdgeConnect/LAMA<br><br>

To run EdgeConnect, move the files of the binary mask, the image with the binary mask overlayed, and the edge map into the folder `edge-connect`.<br><br>
Before running, download the [CelebA Models](https://drive.google.com/drive/folders/13JgMA5sKMYgRwHBp4f7PBc5orNJ_Cv-p) and place it under the `edge-connect/checkpoints` folder. <br><br>
Then run: 
```
python test.py \
  --model 2 \
  --checkpoints ./checkpoints \
  --input [path to file with input image with binary mask overlayed] \
  --mask [path to binary mask file] \
  --output [path to the output directory] \
  --edge [path to edge map]
```
