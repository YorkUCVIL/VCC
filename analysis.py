import copy
import sys

sys.path.append('/home/m2kowal/Research/cc_ace/')
import math
import os
import glob
import json
import pickle
import time
from tqdm import tqdm
import sys
import numpy as np
import argparse
import random

from gen_vcc import plot_classwise_connectome


seed = 0
random.seed(seed)
np.random.seed(seed)

def graph_topology(args):

    cd_paths = glob.glob(args.working_dir + '/*')
    num_graphs = 0
    for path in tqdm(cd_paths):
        if '-' in path:
            continue
        g_path = '{}/connectome/G.pkl'.format(path)
        if args.use_saved_g and os.path.exists(g_path):
            with open(g_path, 'rb') as f:
                G = pickle.load(f)
            itcav_path = '{}/connectome/itcav_scores.json'.format(path)
            with open(itcav_path, 'rb') as f:
                itcavs = json.load(f)
            bn_list = list(itcavs.keys()) + ['class']
        else:
            cd_path = path + '/cd.pkl'
            if not os.path.exists(cd_path):
                continue
            G, bn_list = plot_classwise_connectome(args,
                                          working_dir=path,
                                          statistical_testing=args.statistical_testing,
                                          save_fig=False,
                                          verbose=False)


        if num_graphs == 0:
            # set up dict to track metrics
            topologies = {
                          'branching_factor': [],
                          'branching_factor_per_layer': {bn: [] for bn in bn_list[1:]},
                          'num_concepts_per_layer': {bn: [] for bn in bn_list[:-1]},
                          'num_concepts': [],
                          'itcav_ave': [],
                          'itcav_per_layer_ave': {bn: [] for bn in bn_list[:-1]},
                          'itcav_var': [],
                          'itcav_per_layer_var': {bn: [] for bn in bn_list[:-1]},
                              }

        # number of total concepts in connectome
        topologies['num_concepts'].append(len(G.nodes))

        # number of concepts per layer
        node_quantity = {}
        for bn in bn_list[:-1]:
            concept_count = 0
            for node in G.nodes:
                # if bn in node:
                if node.split(' ')[0] == bn:
                    concept_count += 1
            topologies['num_concepts_per_layer'][bn].append(concept_count)
            node_quantity[bn] = concept_count

        # itcav edge weights
        itcav_scores = []
        itcav_scores_per_lay = {bn: [] for bn in bn_list[:-1]}
        for edge in G.edges:
            itcav = G.get_edge_data(edge[0], edge[1])['weight']
            low_bn = edge[0].split(' ')[0]
            itcav_scores.append(itcav)
            itcav_scores_per_lay[low_bn].append(itcav)
        topologies['itcav_ave'].append(sum(itcav_scores)/len(itcav_scores))
        for bn in bn_list[:-1]:
            try:
                topologies['itcav_per_layer_ave'][bn].append(sum(itcav_scores_per_lay[bn])/len(itcav_scores_per_lay[bn]))
            except:
                continue
        topologies['itcav_var'].append(float(np.var(itcav_scores)))
        for bn in bn_list[:-1]:
            var = float(np.var(itcav_scores_per_lay[bn]))
            # check for nan
            if math.isnan(var):
                continue
            else:
                topologies['itcav_per_layer_var'][bn].append(var)

        # branching factor
        global_branching_factor = []
        for i in range(1, len(bn_list)):
            branches_per_node = []
            for node in G.nodes:
                high_bn = bn_list[i] + ' '
                low_bn = bn_list[i-1] + ' '
                if high_bn in node:
                    # count number of connections to lower layer
                    num_branches = 0
                    for edge in G.edges:
                        if node == edge[1] and low_bn in edge[0]:
                            num_branches += 1
                    branches_per_node.append(num_branches)
                    global_branching_factor.append(num_branches)
            if len(branches_per_node) == 0:
                topologies['branching_factor_per_layer'][bn_list[i]].append(0)
            else:
                topologies['branching_factor_per_layer'][bn_list[i]].append(sum(branches_per_node)/len(branches_per_node))

        if len(global_branching_factor) == 0:
            topologies['branching_factor'].append(0)
        else:
            topologies['branching_factor'].append(sum(global_branching_factor) / len(global_branching_factor))

        num_graphs += 1
        if num_graphs == args.num_classes:
            break

    # summarize metrics and save
    topologies_tmp = copy.deepcopy(topologies)
    for key, value in topologies_tmp.items():
        if isinstance(value, list):
            topologies[key] = sum(value)/ len(value)
        elif isinstance(value, dict):
            for sub_key, sub_value in topologies[key].items():
                try:
                    topologies[key][sub_key] = sum(sub_value)/ len(sub_value)
                except:
                    print(1)
        elif isinstance(value, int):
            topologies[key] = value / num_graphs

    print('----------------------')
    print('----------------------')
    print('Starting results for {} classes | dir {}'.format(num_graphs, args.working_dir))
    print()
    for key, value in topologies.items():
        if isinstance(value, float):
            try:
                print('{} {:.3f}'.format(key, value))
            except:
                print('{} {}'.format(key, value))
        elif isinstance(value, dict):
            print(key)
            for sub_key, sub_value in topologies[key].items():
                try:
                    print('{} {:.3f}'.format(sub_key, sub_value))
                except:
                    print('{} {}'.format(sub_key, sub_value))
        print()
    print('Ending results for {} classes for directory: {}'.format(num_graphs, args.working_dir))
    print('----------------------')
    print('----------------------')

    # save json file
    with open(args.working_dir + '/topologies_{}classes.json'.format(args.num_classes), 'w') as f:
        json.dump(topologies, f)


def parse_arguments(argv):
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()

    # file saving and loading
    parser.add_argument('--working_dir', default='cvpr_outputs/ResNet50_4Lay', type=str,
                        help='Directory to load the pkl file from.')
    parser.add_argument('--graph_postfix', default='G.pkl', type=str,
                        help='Directory to load the pkl file from.')
    parser.add_argument('--use_saved_g', action='store_true',
                        help='If true, use the saved graph metrics, G.pkl instead of re-running the analysis.')

    # analysis
    parser.add_argument('--num_classes', default=10, type=int,
                        help='Number of classes to use per model in analysis')
    parser.add_argument('--statistical_testing', action='store_true',
                        help='If true, perform statistical testing')

    return parser.parse_args(argv)


if __name__ == '__main__':
    start = time.time()
    args = parse_arguments(sys.argv[1:])
    graph_topology(args)
    end = time.time()
    print('Total Time Elapsed: {:.1f} minutes'.format((end - start)/60))



