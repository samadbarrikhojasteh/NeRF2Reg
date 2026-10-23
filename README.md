<h1 align="center">NeRF2Reg</h1>
<h2 align="center"> Registration Neural Radiance Fields, with an Application to Minimally Invasive Surgery </h2>

## 📝About


We propose NeRF2Reg, a novel NeRF architecture which jointly performs registration with a pre-existing 3D model. 
We apply NeRF2Reg to Minimally-Invasive Surgery (MIS), where existing methods perform 3D reconstruction as 
preprocessing, then registration.
## 1. 🚀 Installation and Setup the environment 🚀

### 📦 Dependencies

Since NeRF2Reg is developed upon the Nerfstudio framework and [Nerfacto](https://docs.nerf.studio/nerfology/methods/nerfacto.html), all requirements and dependencies of Nerfstudio remain applicable to MIS-NeRF. 
We have developed and modified version ```v1.0.0``` of Nerfstudio, which can be accessed [here](https://github.com/nerfstudio-project/nerfstudio/releases/tag/v1.0.0).
## 2. 📂 Synthetic liver dataset
This repository includes two synthetic liver experiments: [Synth_Static](Dataset/Synth_Static) and [Synth_Dynamic](Dataset/Synth_Dynamic) consisting of RGB images, depth maps, masks, preoperative model and
corresponding camera poses. 
The RGB images and depth maps were generated using Blender, while the masks were produced using the Segment Anything Model ([SAM](https://github.com/facebookresearch/segment-anything)). 
method.