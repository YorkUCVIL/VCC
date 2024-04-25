import numpy as np
import glob
import time
import argparse
import sys
import pickle
import torch
import matplotlib as mpl
import networkx as nx
import math
import copy
import json
import matplotlib.pyplot as plt
import os
from PIL import Image


def softmax(preds, tau):
    if isinstance(preds, list):
        preds = torch.tensor(preds)
    ex = torch.exp(preds/tau)
    return ex / torch.sum(ex, axis=0)


def plot_classwise_connectome(args,
                              working_dir,
                              statistical_testing=True,
                              save_fig=True,
                              cd=None,
                              alpha=0.3,
                              dpi=110,
                              verbose=True):
    """Plots a visual concept connectome.
    """

    # check if args.concept_img_type exists
    if not hasattr(args, 'concept_img_type'):
        args.concept_img_type = 'overlay_images'
    if not hasattr(args, 'rotate_canvas'):
        args.rotate_canvas = False

    # load concept discovery object
    if cd is None:
        cd_path = working_dir + '/cd.pkl'
        with open(cd_path, 'rb') as f:
          cd = pickle.load(f)
          if verbose:
              print('Loaded cd.pkl from {}'.format(cd_path))

    if len(cd.dic.keys()) >= 8:
        args.edge_cmap = 'binary'
        args.num_concepts_limit = 100
        args.num_img_per_concept = 2
        args.icon_size_factor = 0.00022

    # set parameters
    target_class = cd.target_class
    scores = cd.scores

    if save_fig:
        icon_size_factor = args.icon_size_factor
        num_concepts_limit = args.num_concepts_limit
        num_img_per_concept = args.num_img_per_concept
        edge_thresh = args.edge_thresh
        edge_width = args.edge_width
    else:
        num_img_per_concept = 1
        num_concepts_limit = 25
        edge_thresh = 0
        edge_width = True
        width, height = 1, 1


    # set save dir
    cc_save_dir = working_dir +'/connectome'
    if not os.path.exists(cc_save_dir):
        os.mkdir(cc_save_dir)
    layer_itcav_address = cc_save_dir + '/itcav_scores.json'

    # get layers if none specified
    if save_fig:
        if args.layers_to_show < 1:
            bn_list = cd.bottlenecks.copy()
        else:
            bn_list = cd.bottlenecks.copy()[args.layers_to_show:]
            not_show_bn_list = cd.bottlenecks.copy()[:args.layers_to_show]
    else:
        bn_list = cd.bottlenecks.copy()



    bn_list.append('class')


    # set width based on largest number of concepts in any bn
    if save_fig:
        max_num_concepts = 1
        for bn in bn_list[:-1]:
            concepts = cd.dic[bn]['concepts']
            if len(concepts) > max_num_concepts:
                max_num_concepts = len(concepts)
        if max_num_concepts < num_concepts_limit:
            num_concepts_limit = max_num_concepts
        # set overall figure size based on maximum number of concepts and number of layers
        if num_concepts_limit > 15:
            width, height = (len(bn_list)) * 10, 10 * num_concepts_limit
        else:
            width, height = (len(bn_list)) * 6, 6 * num_concepts_limit

    if save_fig:
        fig = plt.figure(figsize=(width, height), dpi=dpi)


    options = {}


    # create directed graph
    G = nx.DiGraph()
    bn_concepts = {}
    for bn in bn_list[:-1]:
        topk_concepts = cd.dic[bn]['concepts'][:num_concepts_limit]
        bn_concepts[bn] = topk_concepts

    itcav_bn_dict = {bn:[] for bn in bn_list[:-1]}
    nodes_kept = {bn:[] for bn in bn_list}
    for bn, edges in cd.edge_weights.items():
        for edge in edges:
            if 'random' in edge:
                continue
            add_edge = True

            # check if edge is in subset of layers to use
            if save_fig:
                if not args.layers_to_show < 1:
                    for bad_bn in not_show_bn_list:
                        if bad_bn in edge:
                            add_edge = False
                            break
            # get node names
            low_node = edge.split('-')[1]
            low_bn = low_node.split(' ')[0]
            low_concept = low_node.split(' ')[1]
            high_node = edge.split('-')[0]
            high_bn = high_node.split(' ')[0]
            high_concept = high_node.split(' ')[1]

            if statistical_testing:
                if high_bn == 'class':
                    pvalue = cd.do_statistical_testings(cd.scores[low_bn][low_concept], cd.scores[low_bn][cd.random_concept])
                else:
                    pvalue = cd.do_statistical_testings(cd.edge_weights[low_bn][edge], cd.edge_weights[low_bn][edge.split('-')[0]+ '-{} {}'.format(low_bn, cd.random_concept)])
                if pvalue > 0.05:
                    add_edge = False
                    if verbose:
                        print('{} Removed!'.format(edge))

            # set to 0 if it doesn't pass statistical test
            if not add_edge:
                edge_value = 0
            else:
                edge_value = edges[edge]

            # only display if they are in the top-k concepts to show
            if not high_bn == 'class':
                if not high_concept in bn_concepts[high_bn] or not low_concept in bn_concepts[low_bn]:
                    continue
            else:
                if not low_concept in bn_concepts[low_bn]:
                    continue

            # if given in list form, take average
            if isinstance(edge_value, list):
                edge_value = sum(edge_value) / len(edge_value)

            # add edge conditional
            # get img for low_node
            if isinstance(cd.dic[bn][low_node.split(' ')[-1]]['images'], list):
                low_concept_images = np.array([np.load(x) for x in cd.dic[bn][low_node.split(' ')[-1]]['images']])
                low_concept_patches = np.array([np.load(x) for x in cd.dic[bn][low_node.split(' ')[-1]]['patches']])
            else:
                low_concept_images = cd.dic[bn][low_node.split(' ')[-1]]['images']
                low_concept_patches = cd.dic[bn][low_node.split(' ')[-1]]['patches']

            low_concept_img_numbers = cd.dic[bn][low_node.split(' ')[-1]]['image_numbers']
            idxs = np.arange(low_concept_patches.shape[0])[:int(num_img_per_concept**2)]

            low_canvas = np.ones((low_concept_images.shape[2] * num_img_per_concept, low_concept_images.shape[1] * num_img_per_concept, 3))
            # low_canvas = np.ones((low_concept_images.shape[2] * 3, low_concept_images.shape[1] * 3, 3))
            for i, idx in enumerate(idxs):
                row = math.floor(i / num_img_per_concept)
                col = i % num_img_per_concept
                if args.concept_img_type == 'overlay':
                    img = cd.discovery_images[low_concept_img_numbers[idx]]
                    mask = 1 - (np.mean(low_concept_patches[idx] == float(
                        cd.average_image_value) / 255, -1) == 1)
                    # mask = (masks[img_number - 1].permute(1, 2, 0).squeeze() * -1) + 1
                    img_viz = np.where(np.expand_dims(mask,0).repeat(3, 0).transpose(1, 2, 0)  == 0, img*alpha, img)
                elif args.concept_img_type == 'overlay_images':
                    img = cd.discovery_images[low_concept_img_numbers[idx]]
                    mask = 1 - (np.mean(low_concept_patches[idx] == float(
                        cd.average_image_value) / 255, -1) == 1)
                    mask_expanded = np.expand_dims(mask, -1)
                    img_viz_blend = ((mask_expanded * img) + ( (1 - mask_expanded) * img * alpha))
                    ones = np.where(mask == 1)
                    h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
                    img_viz = Image.fromarray((img_viz_blend[h1:h2, w1:w2] * 255).astype(np.uint8))
                    img_viz = np.array(img_viz.resize((224,224), Image.Resampling.BICUBIC)).astype(float) / 255
                elif args.concept_img_type == 'images':
                    img_viz = low_concept_images[idx]
                elif args.concept_img_type == 'patches':
                    img_viz = low_concept_patches[idx]
                else:
                    raise NotImplementedError


                # plot patch
                low_canvas[(row*low_concept_images.shape[2]):(row+1)*low_concept_images.shape[2],
                col*low_concept_images.shape[1]:(col+1)*low_concept_images.shape[1], :] = img_viz

            # get img for high_node
            # check if class
            if high_node.split(' ')[0] == 'class':
                high_canvas = np.ones((low_concept_images.shape[2] * num_img_per_concept, low_concept_images.shape[1] * num_img_per_concept, 3))
                for i in range(int(num_img_per_concept**2)):
                    row = math.floor(i / num_img_per_concept)
                    col = i % num_img_per_concept
                    high_canvas[(row*low_concept_images.shape[2]):(row+1)*low_concept_images.shape[2],
                    col*low_concept_images.shape[1]:(col+1)*low_concept_images.shape[1], :] = cd.discovery_images[i]
            else:
                if isinstance(cd.dic[high_bn][high_node.split(' ')[-1]]['patches'], list):
                    high_concept_images = np.array([np.load(x) for x in cd.dic[high_bn][high_node.split(' ')[-1]]['images']])
                    high_concept_patches = np.array([np.load(x) for x in cd.dic[high_bn][high_node.split(' ')[-1]]['patches']])
                else:
                    high_concept_patches = cd.dic[high_bn][high_node.split(' ')[-1]]['patches']
                    high_concept_images = cd.dic[high_bn][high_node.split(' ')[-1]]['images']
                high_concept_img_numbers = cd.dic[high_bn][high_node.split(' ')[-1]]['image_numbers']
                idxs = np.arange(high_concept_patches.shape[0])[:int(num_img_per_concept**2)]
                high_canvas = np.ones((high_concept_images.shape[2] * num_img_per_concept, high_concept_images.shape[1] * num_img_per_concept, 3))
                for i, idx in enumerate(idxs):
                    row = math.floor(i / num_img_per_concept)
                    col = i % num_img_per_concept
                    if args.concept_img_type == 'overlay':
                        # plot patch 1
                        img = cd.discovery_images[high_concept_img_numbers[idx]]
                        mask = 1 - (np.mean(high_concept_patches[idx] == float(
                            cd.average_image_value) / 255, -1) == 1)
                        img_viz = np.where(np.expand_dims(mask,0).repeat(3, 0).transpose(1, 2, 0)  == 0, img*alpha, img)
                    elif args.concept_img_type == 'overlay_images':
                        img = cd.discovery_images[high_concept_img_numbers[idx]]
                        mask = 1 - (np.mean(high_concept_patches[idx] == float(
                            cd.average_image_value) / 255, -1) == 1)
                        mask_expanded = np.expand_dims(mask, -1)
                        img_viz_blend = ((mask_expanded * img) + ( (1 - mask_expanded) * img * alpha))
                        ones = np.where(mask == 1)
                        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
                        img_viz = Image.fromarray((img_viz_blend[h1:h2, w1:w2] * 255).astype(np.uint8))
                        img_viz = np.array(img_viz.resize((224,224), Image.Resampling.BICUBIC)).astype(float) / 255
                    elif args.concept_img_type == 'images':
                        img_viz = high_concept_images[idx]
                    elif args.concept_img_type == 'patches':
                        img_viz = high_concept_patches[idx]
                    else:
                        raise NotImplementedError

                    high_canvas[(row*high_concept_images.shape[2]):(row+1)*high_concept_images.shape[2],
                            col*high_concept_images.shape[1]:(col+1)*high_concept_images.shape[1], :]  = img_viz

                if args.rotate_canvas:
                    high_canvas = np.rot90(high_canvas, 2)

            itcav_bn_dict[low_bn].append(edge_value)
            if edge_value < edge_thresh:
                continue
            if not low_node in G.nodes:
                nodes_kept[low_bn].append(low_concept)
                G.add_node(low_node, image=low_canvas)
            if not high_node in G.nodes:
                nodes_kept[high_bn].append(high_concept)
                G.add_node(high_node, image=high_canvas)
            if edge_value > 0:
                if edge_width:
                    edge_width = edge_value*10
                    if edge_width < 1:
                        edge_width = 1
                else:
                    edge_width = 5
                G.add_edge(low_node, high_node,
                           weight=edge_value,
                           width=edge_width,
                           alpha=0.7)

    # calculate positions
    pos = {}  # {node: (x, y)
    num_layers = len(bn_list)
    layer_title_pos = []
    for layer_num, bn in enumerate(bn_list):
        sub_title_x_pos = (width / (num_layers + 1)) * (layer_num + 1)
        layer_title_pos.append(sub_title_x_pos)
        # need to only consider the concepts after statistical testing
        num_concepts_in_layer = len(nodes_kept[bn]) if bn != 'class' else 1
        if num_concepts_in_layer > num_concepts_limit:
            num_concepts_in_layer = num_concepts_limit
        concept_num = 0
        for node in G.nodes:
            if bn + ' ' in node:
                node_x_pos = (width / (num_layers + 1)) * (layer_num + 1)
                node_y_pos = (height / (num_concepts_in_layer + 1)) * (concept_num + 1)
                pos[node] = (node_x_pos, node_y_pos)
                concept_num += 1

    # coloring edges
    if save_fig:
        _, edge_colors = zip(*nx.get_edge_attributes(G, 'weight').items())
        cmap = mpl.colormaps[args.edge_cmap]
        nx.draw(G, pos,
                with_labels=False,
                width=[e[2]['width'] for e in G.edges(data=True)],
                edge_color=edge_colors,
                edge_cmap=cmap,
                **options)


        plt.tight_layout()

        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        ax.margins(0.1)


        tr_figure = ax.transData.transform

        # Transform from display to figure coordinates
        tr_axes = fig.transFigure.inverted().transform

        # Select the size of the image (relative to the X axis)
        icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * icon_size_factor
        icon_center = icon_size / 2.0

        # Add the respective image to each node
        for n in G.nodes:
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))
            # get overlapped axes and plot icon
            a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
            if 'class' in n:
                subtitle = n.split(' ')[-1]
                # plot colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
                sm._A = []
                cb = fig.colorbar(sm, shrink=4, pad=0.03, ax=plt.gca())
                cb.ax.tick_params(labelsize=20)
            else:
                subtitle = n.split(' ')[-1].split('_')[-1]
                if args.display_logit_tcav:
                    subtitle = subtitle + ' (TCAV={})'.format(str(round(np.mean(scores[n.split(' ')[0]][n.split(' ')[-1]]), 2)))
            plt.title(subtitle, fontsize=17)
            a.imshow(G.nodes[n]["image"])
            a.axis("off")

        # adding subtitles
        for i, layer in enumerate(bn_list):
            ax.text(layer_title_pos[i], 2.8, layer, fontsize=24, ha="center")

        ax.axis('off')
        plt.axis("off")

        fig.suptitle(working_dir.split('/')[1] + '-' + target_class, fontsize=33, ha="center")

        # save figure
        file_name = '{}/{}'.format(args.working_dir, 'vcc.png')
        print('Saving to {}'.format(file_name))
        fig.savefig(file_name)
        plt.show()
        plt.clf()
        plt.close(fig)

    # save graph for analysis
    with open(layer_itcav_address, "w") as write_file:
        json.dump(itcav_bn_dict, write_file)
    with open(cc_save_dir + '/G.pkl', 'wb') as f:
      pickle.dump(G, f)

    return G, bn_list

def parse_arguments(argv):
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()

    # experiment directory
    parser.add_argument('--working_dir', default='outputs/R50_Zebra', type=str,help='Directory to load the pkl file from.')

    # connectome arguments - change these to adjust the connectome visualization
    parser.add_argument('--num_concepts_limit', default=8, type=int, help="Number of concepts to show per layer")
    parser.add_argument('--num_img_per_concept', default=3, type=int, help="Number of imgs in single row to show per concept (1 | 2 | 3)")
    parser.add_argument('--icon_size_factor', type=float, help="Image icon size", default=0.005)
    parser.add_argument('--edge_cmap', default='plasma_r', type=str,help='Colormap to use for edges. Some good options: plasma_r,cividis_r,bone_r')
    parser.add_argument('--concept_img_type', default='overlay_images', type=str, help="Type of image to display in concepts (images | patches | overlay_images)")
    parser.add_argument('--layers_to_show', type=int, default=-1, help='(Optional), only show the last layers_to_show layers, set to -1 for all layers possible')
    parser.add_argument('--display_logit_tcav', action='store_true', help='If true, display the tcav score in subtitles')
    parser.add_argument('--edge_thresh', default=0, help='Threshold to display edges in graph. Set to 0 to use all')
    parser.add_argument('--edge_width', action='store_false', help='If true, adjust edge widths according to values')

    parser.add_argument('--statistical_testing', action='store_true', help='If true, perform statistical methods before adding edges')
    parser.add_argument('--rotate_canvas', action='store_true', help='If true, rotate the canvas 90 degrees')

    # saving and analysis
    parser.add_argument('--save_layerwise_stats', action='store_false', help='If true, save layerwise stats in json format')
    parser.add_argument('--save_latex', action='store_false', help='If true, save as latex format with separate images')

    parser.add_argument('--model_to_run', type=str,
        help='The name of the pytorch model as in torch hub.', default='vgg11')


    return parser.parse_args(argv)


if __name__ == '__main__':
    start = time.time()
    args = parse_arguments(sys.argv[1:])
    G = plot_classwise_connectome(args, args.working_dir, args.statistical_testing)
    end = time.time()
    print('Total Time Elapsed: {:.1f} minutes'.format((end - start)/60))

