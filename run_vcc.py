"""This script runs the whole VCC method."""
import os
import sys
import time
import shutil
import argparse
import json
import numpy as np

import torch
import random

import vcc_helpers
from vcc import ConceptDiscovery, make_model

# reproducibility
seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def main(args):
    # Create the directories to store the results
    dataset_dir = os.path.join(args.working_dir, 'dataset/')
    dataset_patch_dir = os.path.join(dataset_dir, 'patches/')
    dataset_image_dir = os.path.join(dataset_dir, 'images/')
    discovered_concepts_dir = os.path.join(args.working_dir, 'concepts/')
    results_dir = os.path.join(args.working_dir, 'results/')
    img_mask_dir = os.path.join(args.working_dir, 'image_mask/')
    cavs_dir = os.path.join(args.working_dir, 'cavs/')
    activations_dir = os.path.join(args.working_dir, 'acts/')
    results_summaries_dir = os.path.join(args.working_dir, 'results_summaries/')

    if not os.path.exists(args.working_dir):
        # delete and make directions
        if os.path.exists(args.working_dir):
            shutil.rmtree(args.working_dir)
        os.makedirs(args.working_dir)
        os.makedirs(dataset_dir)
        os.makedirs(dataset_patch_dir)
        os.makedirs(dataset_image_dir)
        # make subdirs
        for bn in args.feature_names:
            os.makedirs(os.path.join(dataset_patch_dir, bn))
            os.makedirs(os.path.join(dataset_image_dir, bn))

        os.makedirs(discovered_concepts_dir)
        os.makedirs(results_dir)
        os.makedirs(img_mask_dir)
        os.makedirs(cavs_dir)
        os.makedirs(activations_dir)
        os.makedirs(results_summaries_dir)

        # save argparse
        with open(results_summaries_dir + '/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    random_concept = 'random_discovery'  # Random concept for statistical testing


    # Creating the ConceptDiscovery class instance
    print('Starting VCC Generation...')

    # if using imagenet, then use the path to entire imagenet dataset
    mymodel = make_model(args)
    # if args.target_dataset == 'imagenet' and len(args.imagenet_path) > 0:
    for key, val in mymodel.class_idx.items():
        if val[1] == args.target_class:
            class_dir = val[0]
    source_dir = '{}/{}'.format(args.imagenet_path, class_dir)

    cd = ConceptDiscovery(
      args,
      mymodel,
      args.target_class,
      random_concept,
      args.feature_names,
      source_dir,
      activations_dir,
      cavs_dir,
      dataset_dir,
      img_shape=args.img_shape,
      num_random_exp=args.num_random_exp,
      channel_mean=args.channel_mean,
      max_imgs=args.max_imgs,
      min_imgs=args.min_imgs,
      num_discovery_imgs=args.num_discovery_imgs,
      correct_incorect=args.correct_incorect,
      save_acts=args.save_acts)


    print('Creating the dataset of feature segments for all layers')
    start = time.time()
    cd.create_patches_top_down(n_top_clusters=int(args.num_segment_clusters[0]),resize_factor=args.resize_factor)
    end = time.time()
    print('Segment dataset creation took {:.2f} minutes'.format((end - start)/60))

    # Saving the concept discovery target class images
    print('Saving the concept discovery target class images')
    image_dir = os.path.join(dataset_dir, 'discovery_images')
    os.makedirs(image_dir)
    vcc_helpers.save_images(image_dir,(cd.discovery_images * 256).astype(np.uint8))

    # Discovering Concepts
    start = time.time()
    print('Discovering concepts')
    if len(args.num_clusters) == 1:
        tmp_num_clusters = []
        for i in range(len(args.feature_names)):
            tmp_num_clusters.append(args.num_clusters[0])
        args.num_clusters = tmp_num_clusters
    num_clusters = {bn: {'n_clusters': int(args.num_clusters[i])} for i, bn in enumerate(args.feature_names)}
    cd.discover_concepts(method=args.clustering_method,
                         param_dicts=num_clusters)
    end = time.time()
    print('Discovering concepts took {:.1f} minutes'.format((end - start)/60))

    # Calculating CAVs and TCAV scores
    print('Calculating CAVS')
    start = time.time()
    cav_accuracies = cd.cavs(min_acc=0.0)
    end = time.time()
    print('Calculating CAVS took {:.1f} minutes'.format((end - start) / 60))

    print('Calculating TCAVS')
    start = time.time()
    cd.scores = cd.target_class_tcavs(test=args.stat_testing_filter)
    end = time.time()
    print('Calculating TCAVS took {:.1f} minutes'.format((end - start) / 60))



    print('Calculating connectome edge weights (ITCAVS)')
    start = time.time()
    cd.edge_weights = vcc_helpers.calc_edge_from_inter_tcav(cd=cd,
                                                tcav_scores=cd.scores,
                                                class_name=args.target_class)

    end = time.time()
    print('Calculating ITCAVS took {:.1f} minutes'.format((end - start) / 60))

    vcc_helpers.save_ace_report(cd, cav_accuracies, cd.scores, results_summaries_dir)

    # cleaning up dirs to save space
    if not args.save_acts:
        shutil.rmtree(activations_dir)
    if not args.save_cavs:
        shutil.rmtree(cavs_dir)

    # clean dataset dir
    vcc_helpers.clean_and_save_cd(args, cd, args.working_dir, dataset_dir)

def parse_arguments(argv):
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()

    # saving args
    parser.add_argument('--working_dir', type=str,
                        help='Directory to save the results_summaries.', default='outputs/R50_Zebra')

    # data args
    parser.add_argument('--random_dir', type=str,
    help='''Directory where the random image folders are saved.''', default='data')
    parser.add_argument('--target_dataset', type=str, default='imagenet',
                        help='Dataset being analyzed. Only works for imagenet currently')
    parser.add_argument('--imagenet_path', type=str, default='/home/m2kowal/data/imagenet/val',
                        help='Path to the imagenet dataset')
    parser.add_argument('--labels_path', type=str,
                        help='Path to imagenet labels json file.', default='./imagenet_class_index.json')

    # model args
    parser.add_argument('--model_to_run', type=str, default='resnet50', help='The name of the model used.'
                             '(resnet50 | vgg16 | tf_mobilenetv3_large_075 | vit_b | mvit | clip_r50).')
    parser.add_argument('--feature_names', nargs='+', default=['layer1', 'layer2', 'layer3', 'layer4'],
        help='Names of the target layers of the network in order from lowest to highest (comma separated)')
    parser.add_argument('--model_path', type=str,
        help='Path to model checkpoint to override original.', default='')
    parser.add_argument('--pretrained', action='store_false',
        help='If true, load pretrained model from torch hub / timm')

    # VCC computation args
    parser.add_argument('--cav_imgs',  type=str,
        help='Type of img to use with cavs (images | patches)', default='images')
    parser.add_argument('--target_class', type=str,
        help='The name of the target class to be interpreted', default='zebra')
    parser.add_argument('--sp_method', type=str, default='MBKM',
        help='The superpixel method used for creating image patches in feature space. (slic | KM | MBKM | DB | HC)')
    parser.add_argument('--mbkm_batch_size', type=int, default=64,
        help='The batch size used for MBKM ')
    parser.add_argument('--correct_incorect', type=str, default='both',
        help='Whether to use correct, incorrect, or both types of predictions. (correct | incorect | correct_incorect)')
    parser.add_argument('--num_segment_clusters', nargs='+', default=[2],
        help="Number of clusters to use for segments, aligned with layers of model. Not used if using elbow method.")
    parser.add_argument('--clustering_method', type=str, default='KM',
        help='The concept clustering method (KM | AP | SC | MS | DB')
    parser.add_argument('--stat_testing_filter', action='store_true',
        help='Whether to filter out concepts via statistical testing. Not used until VCC generation normally')
    parser.add_argument('--num_clusters', nargs='+', default=[25],
        help="Number of concept clusters")
    parser.add_argument('--resize_factor', type=int, default=8,
        help="Factor of image resolution to perform image segment proposal at")
    parser.add_argument('--channel_mean', action='store_false',
        help='if true, average activations across channels')
    parser.add_argument('--use_elbow', action='store_false',
        help='Use elbow method to select num_clusters during top-down segmentation in feature space')
    parser.add_argument('--discovery_n_cluster_method', default='og',
        help='Method to select num_clusters during concept discovery (divide | elbow | og)')
    parser.add_argument('--save_acts', action='store_true',
        help='Flag to save activations')
    parser.add_argument('--save_cavs', action='store_true',
        help='Flag to save tcav .pkl files')
    parser.add_argument('--seg_type', default='images',
        help='type of image to use during activations and itcav (images | patches)')
    parser.add_argument('--img_shape', type=int, default=224, help='')


    # Concept Discovery args
    parser.add_argument('--num_discovery_imgs', type=int,help="Number of discovery images to use",default=50)
    parser.add_argument('--num_random_exp', type=int,help="Number of random experiments used for statistical testing, etc",default=20)
    parser.add_argument('--max_imgs', type=int,help="Maximum number of images in a discovered concept", default=50)
    parser.add_argument('--min_imgs', type=int,help="Minimum number of images in a discovered concept",default=2)

    # Pruning concept clusters
    parser.add_argument('--min_img_method', default='sigmoid',help='method to use to calculate min_imgs (divide | og | root | sigmoid)')
    parser.add_argument('--min_img_root', type=float, default=0.5,help='if min img method is root, the power to use')

    # Generalized sigmoid: https://en.wikipedia.org/wiki/Generalised_logistic_function
    parser.add_argument('--min_img_A', type=float, default=-102, help='Sigmoid parameter A')
    parser.add_argument('--min_img_K', type=float, default=115, help='Sigmoid parameter K')
    parser.add_argument('--min_img_C', type=float, default=1, help='Sigmoid parameter C')
    parser.add_argument('--min_img_Q', type=float, default=1, help='Sigmoid parameter Q')
    parser.add_argument('--min_img_B', type=float, default=0.0004, help='Sigmoid parameter B')
    parser.add_argument('--min_img_v', type=float, default=1, help='Sigmoid parameter v')

    # Parallelization
    parser.add_argument('--cluster_parallel_workers', type=int,help="Number of parallel jobs for clustering.", default=8)

    return parser.parse_args(argv)


if __name__ == '__main__':
    start = time.time()
    main(parse_arguments(sys.argv[1:]))
    end = time.time()
    print('Total Time Elapsed: {:.1f} minutes'.format((end - start)/60))