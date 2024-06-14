import os
import json
import random
random.seed(1)

# a dict of model names and the layers to extract features from
model_layer_dict = {
        'clip_r50': 'layer1 layer2 layer3 layer4',
        'resnet50': 'layer1 layer2 layer3 layer4',
        'vgg16': '8 15 22 29',
        'tf_mobilenetv3_large_075': '0 2 4 6',
        'vit_b': '2 5 8 10',
        'mvit': '1 3 9 15',
}
imagenet_path = '/home/m2kowal/data/imagenet/val'
imagenet_class_index = 'imagenet_class_index.json'
with open(imagenet_class_index, 'r') as f:
    class_idx = json.load(f)
label_list = [class_idx[str(k)][1].replace('-', '_').lower() for k in range(len(class_idx))]
random.shuffle(label_list)

# generate VCCs for the first 10 classes
for model, layers in model_layer_dict.items():
    for cat in label_list[:5]:
        cat = cat.split('/')[-1]
        command = ("python run_vcc.py --imagenet_path {} --model_to_run {} --feature_names {} --target_class {} "
                   "--working_dir outputs/{}_4Lay/{}").format(imagenet_path, model, layers, cat, model, cat)
        cd_save_path = 'outputs/{}_4Lay/{}/cd.pkl'.format(model, cat)
        if not os.path.exists(cd_save_path):
            print(command)
            os.system(command)

    # Perform graph topology analysis on generated VCCs
    os.system('python analysis.py --num_classes 10 --working_dir outputs/{}_4Lay'.format(model))
