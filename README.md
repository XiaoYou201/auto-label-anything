### Quciky start
#### Install segment anything, reference [official repository.](https://github.com/facebookresearch/segment-anything)

The code requires python>=3.8, as well as pytorch>=1.7 and torchvision>=0.8. Please follow the instructions here to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```
The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. jupyter is also required to run the example notebooks.
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```
Then Download segment-anything model and move them to sam directory (if not exist, you can make a new directory named sam in root dir)

default or vit_h: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)  
vit_l: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)  
vit_b: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)  

#### Install tranformers, reference [offcial web.](https://huggingface.co/docs/transformers/en/installation)  
```
pip install transformers
```

When you installed above two model, you can run this program through main function.You can modify args to do that you want.   
Have fun~
