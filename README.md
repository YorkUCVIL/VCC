# Visual Concept Connectome (VCC): Open World Concept Discovery and their Interlayer Connections in Deep Models
[Matthew Kowal](https://mkowal2.github.io/),
[Richard P. Wildes](http://www.cse.yorku.ca/~wildes/), 
[Konstantinos G. Derpanis](https://csprofkgd.github.io/)

Official Implementation of our CVPR 2024 (Highlight) Paper.

[Paper](https://arxiv.org/abs/2404.02233). [Project page](https://yorkucvil.github.io/VCC/), Demo (coming soon)

![AllLayerTeaser](AllLayerTeaser.png)


# Create Conda Environment
```
conda create -n VCC python=3.10.8
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/fvcore'
```

# Data Preparation
- Download ImageNet from http://image-net.org/download
- 20 sets of random images from the Broden dataset are located in data/random* folders

# VCC Generation
To generate VCC for a model, dataset and target class, use the following command and launch run_vcc.py:
```
python run_vcc.py --exp_name outputs/save_path_name --target_class zebra --model model --feature_names layer1 layer2 layer3 layer4 --imagenet_path path_to_imagenet
```

The following models in this table are supported in place of `model` and `feature_names`:

model | 4-layer feature_names | all-layer feature_names
--- | --- | ---
resnet50 | layer1 layer2 layer3 layer4 | layer1.0 layer1.1 layer1.2 layer2.0 layer2.1 layer2.2 layer2.3 layer3.0 layer3.1 layer3.2 layer3.3 layer3.4 layer3.5 layer4.0 layer4.1 layer4.2
vgg16 | 8 15 22 29 | 1 3 6 8 11 13 15 18 20 22 25 27 29
tf_mobilenetv3_large_075 | 0 2 4 6 | 0.0 1.0 1.1 2.0 2.1 2.2 3.0 3.1 3.2 3.3 4.0 4.1 5.0 5.1 5.2 6.0
vit_b | 2 5 8 10 | 0 1 2 3 4 5 6 7 8 9 10
mvit | 1 3 9 16 | 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
clip_r50 | layer1 layer2 layer3 layer4 | layer1.0 layer1.1 layer1.2 layer2.0 layer2.1 layer2.2 layer2.3 layer3.0 layer3.1 layer3.2 layer3.3 layer3.4 layer3.5 layer4.0 layer4.1 layer4.2

The checkpoint for the MViT model can be found [here](https://drive.google.com/file/d/15oRrMvKv7v9GFGqLs64sF2dsndMzdsew/view?usp=sharing).

# VCC Visualization
To visualize and save VCCs, use the following command:
```
python gen_vcc.py --working_dir outputs/save_path_name
```


# VCC Analysis
To compute graph metrics averaged over VCCs, use analysis.py. To compute VCCs for 10 randomly selected ImageNet classes, use the following command:
```
python exps/10VCC_analysis.py
```

# Citation
If you find this work useful, please consider citing:
```
@inproceedings{kowal2024visual,
  title={Visual Concept Connectome (VCC): Open World Concept Discovery and their Interlayer Connections in Deep Models},
  author={Kowal, Matthew and Wildes, Richard P and Derpanis, Konstantinos G},
  booktitle={Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```


# Acknowledgements
Code structure modified from the [ACE](https://github.com/amiratag/ACE) repository.