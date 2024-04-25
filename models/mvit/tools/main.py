# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Wrapper to train/test models."""

import argparse
import sys

import models.mvit.mvit.utils.checkpoint as cu
from models.mvit.tools.engine import test, train
from models.mvit.mvit.config.defaults import assert_and_infer_cfg, get_cfg
from models.mvit.mvit.utils.misc import launch_job


def parse_args():
    """
    Parse the following arguments for a default parser.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/MVIT_B.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See mvit/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )


    parser.add_argument('--source_dir', type=str,
    help='''Directory where the network's classes image folders and random
    concept folders are saved.''', default='data/data_tcav')
    parser.add_argument('--working_dir', type=str,
        help='Directory to save the results_summaries.', default='outputs/test3')
    parser.add_argument('--model_to_run', type=str,
        # help='The name of the pytorch model as in torch hub.', default='googlenet')
        help='The name of the pytorch model as in torch hub.', default='resnet50')
    parser.add_argument('--model_path', type=str,
        help='Path to model checkpoints.', default='')
    parser.add_argument('--cluster_template_path', default='', type=str,
    # parser.add_argument('--cluster_template_path', default='data/concept_template/googlenet_broden224_masked_max100', type=str,
        help='Path to model checkpoints.')
    parser.add_argument('--labels_path',  type=str,
        help='Path to model checkpoints.', default='./imagenet_class_index.json')
    parser.add_argument('--perturb_order', type=str, default='random',
                        help='type of perturbing to do (normal | random | reverse)')
    parser.add_argument('--perturb_mode', type=str, default='ssc',
                        help='Whether to add the single concept or remove it')
    parser.add_argument('--use_slic_transformers', action='store_true',
                        help='Whether to use SLIC for the first round of clustering for transformers')
    parser.add_argument('--target_class', type=str,
        help='The name of the target class to be interpreted', default='dumbbell')
    parser.add_argument('--imagenet_path', type=str, default='/home/m2kowal/data/imagenet/val' ,help='')
    parser.add_argument('--sp_method', type=str, default='KM',
        help='The superpixel method used for creating image patches. (slic | KM | DB | HC)')
    parser.add_argument('--num_segment_clusters', nargs='+', default=[2],
        help="Number of clusters to use for segments, aligned with layers of model")
    parser.add_argument('--clustering_method', type=str, default='KM',
        help='The clustering method (KM | AP | SC | MS | DB')
    parser.add_argument('--stat_testing_filter', action='store_true',
        help='Whether to filter out concepts via statistical testing')
    parser.add_argument('--num_clusters', type=int,
        help="Number of clusters", default=10)
    parser.add_argument('--num_masks', type=int,
        help="Number of masks", default=10)
    parser.add_argument('--random_dir', type=str,
    help='''Directory where the random concept folders are saved.''', default='data/data_tcav')
    parser.add_argument('--channel_mean', action='store_false',
        help='if true, average activations across channels')
    parser.add_argument('--itcav_use_patch', action='store_true',
        help='if true, use non-resize patches for itcav calculations')
    parser.add_argument('--cluster_dot', action='store_true',
        help='if true, dot product with cluster center during edge weight calculation')
    parser.add_argument('--combine_cluster_thresh', type=float, default=0,
        help='Threshold to collapse cluster centers with, set to 0 if you want no collapse')
    parser.add_argument('--save_acts', action='store_true',
        help='Flag to save activations')
    parser.add_argument('--save_cavs', action='store_true',
        help='Flag to save tcav .pkl files')
    parser.add_argument('--resize_factor', type=int, default=8,
        help="Factor of image resolution to perform image segment proposal at")
    # parser.add_argument('--feature_names', nargs='+', default=['1', '3', '9', '15'],
    parser.add_argument('--itcav_gip_type', type=str, default='max', help='What concpet to grab based on the GIP value (max | min)')
    parser.add_argument('--feature_names', nargs='+', default=['layer1', 'layer2','layer3', 'layer4'],
    # parser.add_argument('--feature_names', nargs='+', default=['inception4c', 'inception5b'],
    # parser.add_argument('--feature_names', nargs='+', default=['4', '7', '9', '11'],
        help='Names of the target layers of the network in order from lowest to highest (comma separated)')
    parser.add_argument('--perturb_dir', type=str, default='inverse',
                        help='Direction of perturbation (target | random | inverse)')
    parser.add_argument('--perturb_eps', nargs='+', default=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
                        help='Type of score func for perturb score (acc | logit | softmax | norm)')
    parser.add_argument('--num_random_exp', type=int,
        help="Number of random experiments used for statistical testing, etc",
                      # default=20)
                      default=4)
    parser.add_argument('--num_cat_to_use', default=10, type=int, help="Number of classes to use")

    parser.add_argument('--max_imgs', type=int,
        help="Maximum number of images in a discovered concept",
                      default=20)
                      # default=12)
    parser.add_argument('--min_imgs', type=int,
        help="Minimum number of images in a discovered concept",
                      default=10)
                      # default=3)
    parser.add_argument('--min_img_root', type=float, default=0.5,
                        help='method to use to calculate min_imgs (divide | og | root)')
    parser.add_argument('--min_img_method', default='images',
                        help='type of image to use during activations and itcav (images | patches)')
    parser.add_argument('--discovery_n_cluster_method', default='og',
        help='Method to select num_clusters during concept discovery (divide | elbow | og)')
    parser.add_argument('--num_parallel_workers', type=int,
        help="Number of parallel jobs.",default=0)
    parser.add_argument('--use_ace_seg', action='store_true',
        help='Whether to use the OG ACE segmentation patch')
    parser.add_argument('--save_file', default='ssc_gip_patches', type=str,help='Directory to save the results_summaries.')
    parser.add_argument('--json_file', default='ssc_gip_patches', type=str,help='Directory to save the results_summaries.')

    parser.add_argument('--use_train', type=bool, default=False, help='If true, use separate training and validation connectomes')
    parser.add_argument('--plot_type', type=str, default='single', help='If true, plot all layers on single plot')
    parser.add_argument('--image_type', type=str, default='patches', help='Type of images to use for peturb input (images | patches)')
    parser.add_argument('--per_score_accum_type', type=str, default='sum', help='Method to accumulate the different edges in a single layer (product | sum)')
    parser.add_argument('--global_accum_type', type=str, default='mean', help='Method to accumulate the different layers (max | mean)')
    parser.add_argument('--scoring_func', type=str, default='logit', help='Type of score func for perturb score (acc | logit | softmax | norm)')
    parser.add_argument('--num_discovery_imgs', type=int,help="Number of discovery images to use",default=50)

    parser.add_argument('--working_dir_val', default='outputs/googlenet_ImageNet100_elbow_final1', type=str,
        help='Directory to save the results_summaries.')
    parser.add_argument('--layer_idx_to_perturb', nargs='+', default=[0,1,2,3], help='Type of score func for perturb score (acc | logit | softmax | norm)')
    parser.add_argument('--save_imgs', action='store_true',
        help='Flag to save img files')
    parser.add_argument('--seg_type', default='images',
        help='type of image to use during activations and itcav (images | patches)')
    parser.add_argument('--resume', action='store_true',
        help='resume from where the run left off (focus on segment proposal part)')

    parser.add_argument('--min_img_A', type=float, default=1, help='')
    parser.add_argument('--min_img_K', type=float, default=550, help='')
    parser.add_argument('--min_img_C', type=float, default=1.5, help='')
    parser.add_argument('--min_img_Q', type=float, default=10, help='')
    parser.add_argument('--min_img_B', type=float, default=0.0014, help='')
    parser.add_argument('--min_img_v', type=float, default=0.5, help='')
    parser.add_argument('--cluster_parallel_workers', type=int,
        help="Number of parallel jobs for clustering.",default=0)
    parser.add_argument('--mbkm_batch_size', type=int, default=1024,
        help='The batch size used for MBKM ')

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    # cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
