# Variational Shape Learner

This repository contains the source codes for the paper: [Learning a Hierarchical Latent-Variable Model of Voxelized 3D Shapes](https://arxiv.org/abs/1705.05994), introduced by [Shikun Liu](http://shikun.io/),  [Alexander G. Ororbia II](http://www.personal.psu.edu/ago109/), [C. Lee Giles](https://clgiles.ist.psu.edu/).

<img src="plots/vis_1.png"  width="100px"/><img src="plots/vis_2.png"  width="100px"/><img src="plots/vis_3.png"  width="100px"/>
<img src="plots/vis_4.png"  width="80px"/><img src="plots/vis_5.png"  width="100px"/><img src="plots/vis_6.png"  width="100px"/><img src="plots/vis_7.png"  width="100px"/><img src="plots/vis_8.png"  width="100px"/><img src="plots/vis_9.png"  width="80px"/>

## Requirements
VSL was written in `python 3.6`. For running the code, please make sure the following packages have been installed.
- h5py 2.7
- matplotlib 1.5
- mayavi 4.5
- numpy 1.12
- scikit-learn 0.18
- tensorflow 1.0

Most of which can be directly installed using `pip` command. However, `mayavi` which is used for 3D voxel visualization is recommended to be installed using `conda` enviroment (for simplicity).

## Dataset
We use both 3D shapes from [ModelNet](http://modelnet.cs.princeton.edu/) and [PASCAL 3D+ v1.0](http://cvgl.stanford.edu/projects/pascal3d.html) aligned with images in [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) for training VSL. ModeNet is used for general 3D shape learning including shape geneartion, interpolation and classification. PASCAL 3D is only used for image reconsturction.

Please download the dataset here: [[link]](https://www.dropbox.com/s/sk756qif5tfk9w3/dataset.zip?dl=0).

The above dataset contains files `ModelNet10_res30_raw.mat` and `ModelNet40_res30_raw.mat` representing voxelized version of Modelnet10/40 and  `PASCAL3D.mat` which represents voxelized PASCAL3D+ aligned with images.

Each ModelNet dataset contains `train` and `test` split with each entry has `270001` dimension representing `[id|voxel]` in `[30x30x30]` resolution.

PASCAL3D contains `image_train`, `model_train`, `image_test`, `model_test` which were defined in [Kar, et al](https://github.com/akar43/CategoryShapes). Each entry of `model` again is in `270001` dimension which is similar defined in ModelNet and each entry of `image` is in `[100,100,3]` dimension representing [100x100] RGB images.

## Parameters
We have also included the pre-trained model for parameters can be downloaded [here](https://www.dropbox.com/s/pz5kqi8guq0jxgm/parameters.zip?dl=0).

## Training VSL
Please download `dataset` and `parameters` (if using pre-trained parameters) from links in the previous sections and extract them in the same folder of this repository.

Please use `vsl_main.py` for general 3D shape learning experiments, and `vsl_imrec.py` for image reconstruction experiment. For correctly using the hyper-parameters in the pre-trained model and consistent with the other experiment settings in the paper, please define hyper-parameters as follows,

||ModelNet40 | ModelNet10 | PASCAL3D (jointly) | PASCAL3D (separtely)|
|---|---|---|---|---|
`global_latent_dim` | 20 | 10|10|5|
`local_latent_dim` | 10 | 5|5|2|
`local_latent_num` | 5 | 5|5|3|
`batch_size` | 200 | 10 | 40 | 5|

The source codes are fully commented. For any more details please look over the paper and source code.

Normally, training VSL from scratch requires 2 days on ModelNet in a fast computer, and requires 10-20 minutes on separtely-trained image reconstruction experiment.


## Citation
If you found this work is useful for your research, please consiering cite:

```
@article{liu2017learning,
  title={Learning a Hierarchical Latent-Variable Model of Voxelized 3D Shapes},
  author={Liu, Shikun and Ororbia, II and Alexander, G and Giles, C Lee},
  journal={arXiv preprint arXiv:1705.05994},
  year={2017}
}
```

## Contact
If you found any questions, please contact `sk.lorenmt@gmail.com` for more details.
