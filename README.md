# LVS-Net: Line Voting with Segmentation for Visual Localization

### Introduction
This work is the extension of VS-Net for line segment landmarks. The code is built upon [VS-Net](https://github.com/zju3dv/VS-Net). 

Briefly, we trained the deep NN to predict both the line segmentation map and the [attraction field map](https://github.com/cherubicXN/afm_cvpr2019) from input camera image. We tested our model on Cambridge Landmark Dataset and achieved similar accuracy with VS-Net on 3 out of 5 scenes. Future work includes the improvement of 3D line segment quality and the integration of point and line landmarks for establishing 2D-3D correspondences.

### Requirements
We recommend Ubuntu 18.04, cuda 10.0, python 3.7.
```
conda env create -f environment.yaml
conda activate lvsnet
cd scripts
sh environment.sh
cd squeeze
python setup.py build_ext --inplace
```
In order to test the model, you also need to install Python bindings of [PoseLib](https://github.com/vlarsson/PoseLib/tree/master/pybind).

### Data
For reimplementation convenice, we share the pre-processed [data](https://drive.google.com/file/d/1DwMvz4yjM9ajMoIaRC8ZDk1SReT8XAlO/view?usp=sharing) on KingsCollege. You can also download the raw Cambrdige Landmarks Dataset from [here](http://mi.eng.cam.ac.uk/projects/relocalisation).
```
python scripts/cambridge_line_preprocess.py --scene scene_name --root_dir /path/to/data
```

### Training
```
python train.py --dataset {dataset}_loc --scene {scene_name} --use-aug true --gpu-id gpu_device_id --experiment {experiment_name}
```

### Evaluation
We provide the pretrained model for KingsCollege [here](https://drive.google.com/file/d/1hcVN9Ed9dCyu5ECUIDur0c6wGnxTzPN1/view?usp=sharing).
```
python test_all.py
```

### Acknowledgements
Thanks Rémi Pautrat and Shaohui Liu for their guidance and supervision.
