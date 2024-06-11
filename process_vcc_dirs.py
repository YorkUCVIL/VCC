import os
import pickle
import math
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# dir structure: parent_vcc_dir -> model_dir -> class_dir -> cd.pkl
parent_vcc_dir = 'demo_outputs'
save_dir = 'demo_outputs_processed'

print('Saving processed data to:', save_dir)

# make save_dir if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# iterate over all model directories
for model_dir in tqdm(os.listdir(parent_vcc_dir)):
    model_demo_dir = os.path.join(parent_vcc_dir, model_dir)
    if not os.path.isdir(model_demo_dir):
        continue

    print('Processing model:', model_dir)
    # iterate over all class directories
    for class_dir in os.listdir(model_demo_dir):
        class_demo_dir = os.path.join(model_demo_dir, class_dir)
        if not os.path.isdir(class_demo_dir):
            continue

        # iterate over all pickle files
        pickle_file = 'cd.pkl'

        pickle_file = os.path.join(class_demo_dir, pickle_file)
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                    cd = pickle.load(f)

            vcc_dic = {}
            vcc_dic['edge_weights'] = cd.edge_weights
            # average the edge weights
            for k, v in vcc_dic['edge_weights'].items():
                vcc_dic['edge_weights'][k] = {k2: np.mean(v2) for k2, v2 in v.items()}
            vcc_dic['images'] = {}
            vcc_dic['size_spec'] = [len(cd.dic[bn]['concepts']) for bn in cd.bottlenecks] + [1]
            vcc_dic['layers'] = cd.bottlenecks + ['class']

            # create new directory for processed data
            processed_dir = os.path.join(save_dir, model_dir, class_dir)
            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)

            print('processing class:', class_dir)

            for concept_layer in cd.bottlenecks + ['class']:
                processed_layer_dir = os.path.join(processed_dir, concept_layer)
                if not os.path.exists(processed_layer_dir):
                    os.makedirs(processed_layer_dir)

                if concept_layer.split(' ')[0] == 'class':
                    vcc_dic['images'][concept_layer] = [os.path.join(processed_dir, concept_layer, 'class')]
                    concept_name = 'class'
                    canvas = np.ones((224 * 3,
                                           224 * 3, 3))
                    for i in range(int(3 ** 2)):
                        row = math.floor(i / 3)
                        col = i % 3
                        canvas[(row * 224):(row + 1) * 224,
                        col * 224:(col + 1) * 224, :] = \
                        cd.discovery_images[i]
                else:
                    vcc_dic['images'][concept_layer] = [os.path.join(processed_dir, concept_layer, concept) for concept
                                                        in cd.dic[concept_layer]['concepts']]
                    for concept_name in cd.dic[concept_layer]['concepts']:

                        # get img for low_node
                        if isinstance(cd.dic[concept_layer][concept_name]['images'], list):
                            low_concept_images = np.array(
                                [np.load(x) for x in cd.dic[concept_layer][concept_name]['images']])
                            low_concept_patches = np.array(
                                [np.load(x) for x in cd.dic[concept_layer][concept_name]['patches']])
                        else:
                            low_concept_images = cd.dic[concept_layer][concept_name]['images']
                            low_concept_patches = cd.dic[concept_layer][concept_name]['patches']

                        low_concept_img_numbers = cd.dic[concept_layer][concept_name]['image_numbers']
                        idxs = np.arange(low_concept_patches.shape[0])[:int(3 ** 2)]

                        canvas = np.ones((low_concept_images.shape[2] * 3,
                                              low_concept_images.shape[1] * 3, 3))
                        for i, idx in enumerate(idxs):
                            row = math.floor(i / 3)
                            col = i % 3

                            img = cd.discovery_images[low_concept_img_numbers[idx]]
                            mask = 1 - (np.mean(low_concept_patches[idx] == float(
                                cd.average_image_value) / 255, -1) == 1)
                            mask_expanded = np.expand_dims(mask, -1)
                            img_viz_blend = ((mask_expanded * img) + ((1 - mask_expanded) * img * 0.3))
                            ones = np.where(mask == 1)
                            h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
                            img_viz = Image.fromarray((img_viz_blend[h1:h2, w1:w2] * 255).astype(np.uint8))
                            img_viz = np.array(img_viz.resize((224, 224), Image.Resampling.BICUBIC)).astype(
                                float) / 255

                            # plot patch
                            canvas[(row * low_concept_images.shape[2]):(row + 1) * low_concept_images.shape[2],
                            col * low_concept_images.shape[1]:(col + 1) * low_concept_images.shape[1], :] = img_viz

                        # show image with matplotlib
                        # plt.imshow(canvas)
                        # plt.show()

                # save image
                canvas = Image.fromarray((canvas * 255).astype(np.uint8))
                canvas.save(os.path.join(processed_layer_dir, f'{concept_name}.png'))


            # save json file
            vcc_dic_save_path = os.path.join(processed_dir, 'vcc_info.json')
            with open(vcc_dic_save_path, 'w') as f:
                json.dump(vcc_dic, f)
print('Done!')