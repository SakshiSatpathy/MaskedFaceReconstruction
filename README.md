# MaskedFaceReconstruction
### Pipeline:
1. sam_experiment.py:<br>
This code allows the user to segment the original images using SAM.
Note: you have to manually update the filepaths within the code.
Running the code will prompt a small GUI to select points to include/exclude in the segmented image.
Left-clicking on a point (shown as a blue dot) will INCLUDE that point, and Right-Clicking (shown as a red dot) will EXCLUDE that point.

2. 3DMM-Fitting-Pytorch: <br>
Forked from [this repository](https://github.com/ascust/3DMM-Fitting-Pytorch/tree/master)
To run the code, you must download the [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads) and place "01_MorphableModel.mat" into the folder "3DMM-Fitting-Pytorch/BFM"
You must also download the Expression Basis. Go to the [repo](https://github.com/Juyong/3DFace), download the "CoarseData" and put "Exp_Pca.bin" into "BFM".
After that, you must convert the BFM parameters by running `python convert_bfm09_data.py` under "3DMM-Fitting-Pytorch".
