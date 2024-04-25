import gc
import glob
import copy
import json
import scipy.stats as stats
import skimage.segmentation as segmentation
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as metrics
from skimage.transform import resize
from yellowbrick.cluster.elbow import kelbow_visualizer
import torch
import torchvision
from collections import OrderedDict
from tcav import cav
from vcc_helpers import *
import models.clip as clip
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
class ConceptDiscovery(object):
  def __init__(self,
               args,
               model,
               target_class,
               random_concept,
               bottlenecks,
               source_dir,
               activation_dir,
               cav_dir,
               dataset_dir,
               img_shape=224,
               num_random_exp=2,
               channel_mean=True,
               cluster_dot=True,
               correct_incorect='both',
               max_imgs=40,
               min_imgs=20,
               num_discovery_imgs=40,
               average_image_value=117,
               save_acts=True,
               bs=8):
    """Generates a Visual Concept Connectome (VCC) for a given class in a trained model.

    For a trained classification model, the ConceptDiscovery class
    performs unsupervised concept discovery using examples of one of the classes
    in the network and then computes the interlayer edge weight contributions between them.
    """
    self.args = args
    self.model = model
    self.target_class = target_class
    self.num_random_exp = num_random_exp
    if isinstance(bottlenecks, str):
      bottlenecks = [bottlenecks]
    self.bottlenecks = bottlenecks
    self.source_dir = source_dir
    self.activation_dir = activation_dir
    self.dataset_dir = dataset_dir
    self.cav_dir = cav_dir
    self.channel_mean = channel_mean
    self.cluster_dot = cluster_dot
    self.random_concept = random_concept
    self.image_shape = [img_shape, img_shape]
    if 'mobilenet' in self.args.model_to_run:
      self.mean = [0.5, 0.5, 0.5]
      self.std = [0.5, 0.5, 0.5]
    else:
      self.mean = [0.485, 0.456, 0.406]
      self.std = [0.229, 0.224, 0.225]
    self.save_acts = save_acts
    self.max_imgs = max_imgs
    self.min_imgs = min_imgs
    self.correct_incorect = correct_incorect
    self.bs = bs
    if num_discovery_imgs is None:
      num_discovery_imgs = max_imgs
    self.num_discovery_imgs = num_discovery_imgs
    self.average_image_value = average_image_value

  def load_concept_imgs(self, concept, max_imgs=1000, source_dir=None):
    """Loads all colored images of a concept.

    Args:
      concept: The name of the concept to be loaded
      max_imgs: maximum number of images to be loaded

    Returns:
      Images of the desired concept or class.
    """
    if source_dir is None:
      concept_dir = self.source_dir
    else:
      concept_dir = os.path.join(source_dir, concept)
    img_paths = glob.glob(concept_dir + '/*')
    if len(img_paths) == 0:
      print('No target images found in {}'.format(concept_dir))
      raise FileNotFoundError
    return load_images_from_files(
        img_paths,
        max_imgs=max_imgs,
        return_filenames=False,
        do_shuffle=False,
        shape=(self.image_shape))

  def load_correct_incorrect_imgs(self, concept, max_imgs=1000, source_dir=None):
    """Loads all colored images of a concept.

    Args:
      concept: The name of the concept to be loaded
      max_imgs: maximum number of images to be loaded

    Returns:
      Images of the desired concept or class based on prediction accuracy.
    """
    if source_dir is None:
      concept_dir = self.source_dir
    else:
      concept_dir = os.path.join(source_dir, concept)
    img_paths = glob.glob(concept_dir + '/*')
    if len(img_paths) == 0:
      print('No target images found in {}'.format(concept_dir))
      raise FileNotFoundError
    final_img_list = []

    print('Trying to find {} {} images...'.format(self.num_discovery_imgs, self.correct_incorect))

    class_id = self.model.label_to_id[self.target_class.replace('_', ' ').replace('-', ' ').lower()]
    for img_path in tqdm(img_paths):
      img = load_image_from_file(img_path, self.image_shape)
      if img is None:
        # deal with corrupt training images...
        continue
      img_tensor = torchvision.transforms.functional.normalize(
                               torch.tensor(img).unsqueeze(0).cuda().permute(0, 3, 1, 2),
                               mean=self.mean, std=self.std).float()
      if 'clip' in self.args.model_to_run:
        image_features = self.model.encode_image(img_tensor)
        # normalized features
        image_features = (image_features / image_features.norm(dim=-1, keepdim=True)).type(torch.float32)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp().type(torch.float32)
        outputs = logit_scale * image_features @ self.model.text_features.t()
      else:
        outputs = self.model(img_tensor)
      for bn in self.bottlenecks:
        features_blobs.pop(0)

      pred_idx = outputs.argmax().item()

      if pred_idx == class_id:
        if self.correct_incorect == 'correct':
          final_img_list.append(img)
      else:
        if self.correct_incorect == 'incorrect':
          final_img_list.append(img)

      if len(final_img_list) == self.num_discovery_imgs:
        break

    print('Found {} images predicted {}'.format(len(final_img_list), self.correct_incorect))
    return np.stack(final_img_list)

  def create_patches_top_down(self, resize_factor=8, discovery_images=None, n_top_clusters=2):
    """Creates a set of image patches using superpixel methods on the feature activations.
    """

    # load images
    if discovery_images is None:
        if self.correct_incorect == 'both':
          raw_imgs = self.load_concept_imgs(self.target_class, self.num_discovery_imgs)
        else:
          raw_imgs = self.load_correct_incorrect_imgs(self.target_class, self.num_discovery_imgs)
        self.discovery_images = raw_imgs
    else:
        self.discovery_images = discovery_images

    # batch images
    output = {bn: [] for bn in self.bottlenecks}
    for i in range(int(self.discovery_images.shape[0] / self.bs) + 1):
      img_batch = self.discovery_images[i * self.bs:(i + 1) * self.bs]
      if img_batch.shape[0] == 0:
        break
      img_batch_tensor = torchvision.transforms.functional.normalize(
                               torch.tensor(img_batch).cuda().permute(0, 3, 1, 2),
                               mean=self.mean, std=self.std).float()
      if 'clip' in self.args.model_to_run:
        _ = self.model.encode_image(img_batch_tensor)
      else:
        _ = self.model(img_batch_tensor)
      for i, bn in enumerate(self.bottlenecks):
        out = features_blobs[0].transpose(0,2,3,1)
        output[bn].append(out)
        features_blobs.pop(0)
        if self.args.model_to_run.find('vit') > -1 and not self.args.model_to_run == 'mvit':
          # reset cls token collector since we don't need right meow
          cls_token_blobs.pop(0)

    bn_shapes = {}
    for bn in self.bottlenecks:
      bn_shapes[bn] = output[bn][0].shape[1:]

    bn_dataset = {bn: [] for bn in self.bottlenecks}
    bn_patches = {bn: [] for bn in self.bottlenecks}
    bn_image_numbers = {bn: [] for bn in self.bottlenecks}
    bn_dataset_mask_assignment = {bn: [] for bn in self.bottlenecks}

    if self.args.use_elbow:
      self.elbow_vals = {bn: [] for bn in self.bottlenecks}

    for bn in self.bottlenecks:
      discovery_acts = np.concatenate(output[bn], axis=0)
      discovery_acts = np.array(torch.nn.functional.interpolate(torch.tensor(discovery_acts.transpose(0, 3, 1, 2)).type(torch.float32),
                                                                size=(int(img_batch.shape[1] / resize_factor),
                                                                      int(img_batch.shape[2] / resize_factor)),
                                                                mode='bilinear')).transpose(0, 2, 3, 1)
      output[bn] = discovery_acts


    # hierarchical activation segmentation
    for fn in tqdm(range(len(output[bn]))):
      # create list of all activations
      single_img_acts = {layer: output[layer][fn] for layer in output.keys()}

      bn_image_superpixels, bn_image_patches, bn_mask_assignment, _ = self._return_top_down_img_segments(
        img=self.discovery_images[fn],
        single_img_acts=single_img_acts,
        n_top_clusters=n_top_clusters,
        bn_shapes=bn_shapes)

      for bn in self.bottlenecks:
        for seg_idx in range(len(bn_image_patches[bn])):
          image_superpixels = bn_image_superpixels[bn][seg_idx]
          image_patches = bn_image_patches[bn][seg_idx]
          assignment = bn_mask_assignment[bn][seg_idx]

          # set up paths and file names
          file_name = 'seg_{}_{}.npy'.format(fn, seg_idx)
          image_path = os.path.join(self.dataset_dir, 'images', bn, file_name)
          patches_path = os.path.join(self.dataset_dir, 'patches', bn, file_name)

          # save np files
          np.save(image_path, image_superpixels)
          np.save(patches_path, image_patches)

          bn_dataset[bn].append(image_path)
          bn_patches[bn].append(patches_path)
          bn_image_numbers[bn].append(fn)
          bn_dataset_mask_assignment[bn].append(assignment)

    # collapse to numpy array
    for bn in self.bottlenecks:
      bn_dataset[bn] = bn_dataset[bn]
      bn_patches[bn] = bn_patches[bn]
      bn_image_numbers[bn] = np.array(bn_image_numbers[bn])
      bn_dataset_mask_assignment[bn] = np.array(bn_dataset_mask_assignment[bn])

    self.bn_dataset, self.bn_image_numbers, self.bn_patches, self.bn_dataset_mask_assignment =\
    bn_dataset, bn_image_numbers, bn_patches, bn_dataset_mask_assignment

  def _return_top_down_img_segments(self, img, single_img_acts, n_top_clusters=2,bn_shapes=None,compactness=0.8,save_bg='mean'):

    masks = {bn: [] for bn in single_img_acts.keys()}
    mask_assignment = {bn: [] for bn in single_img_acts.keys()}
    mask_labels = {bn: [] for bn in single_img_acts.keys()}
    bn_list = [bn for bn in single_img_acts.keys()]
    # reverse list to get images in top down manner
    bn_list.reverse()
    for i, bn in enumerate(bn_list):
      act = single_img_acts[bn]

      # first cluster
      if i == 0:
        if self.args.use_elbow:
          # use elbow method to determine the number of clusters
          if self.args.sp_method == 'MBKM':
            clust = cluster.MiniBatchKMeans(batch_size=self.args.mbkm_batch_size)
          else:
            clust = cluster.KMeans()

          elbow_alg = kelbow_visualizer(clust,
                                        act.reshape(-1, act.shape[-1]),
                                        k=3,
                                        metric='silhouette',
                                        show=False,
                                        timings=False,
                                        )
          if self.args.cluster_parallel_workers > 0:
            with threadpool_limits(user_api="openmp", limits=self.args.cluster_parallel_workers):
              elbow_alg.fit(act.reshape(-1, act.shape[-1]))
          else:
            elbow_alg.fit(act.reshape(-1, act.shape[-1]))
          # close plots
          plt.clf()
          plt.close()
          n_top_clusters = elbow_alg.elbow_value_
          self.elbow_vals[bn].append(n_top_clusters)
          if n_top_clusters == None:
            n_top_clusters = 2

        # define kmeans algorithm
        if self.args.cluster_parallel_workers > 0:
          with threadpool_limits(user_api="openmp", limits=self.args.cluster_parallel_workers):
            if self.args.sp_method == 'MBKM':
              alg = cluster.MiniBatchKMeans(n_clusters=n_top_clusters,batch_size=self.args.mbkm_batch_size)
            else:
              alg = cluster.KMeans(n_clusters=n_top_clusters)
        else:
          if self.args.sp_method == 'MBKM':
            alg = cluster.MiniBatchKMeans(n_clusters=n_top_clusters, batch_size=self.args.mbkm_batch_size)
          else:
            alg = cluster.KMeans(n_clusters=n_top_clusters)
        labels = alg.fit_predict(act.reshape(-1, act.shape[-1]))
        label_nums = np.unique(labels)
        segments = labels.reshape(act.shape[0], act.shape[1])

        for s in range(label_nums.max() + 1):
          mask = np.where(segments == s,1,0)
          if mask.mean() > 0.001:
            masks[bn].append(mask)
            # set all top layer segments to 0
            mask_assignment[bn].append(0)
            mask_labels[bn].append(s)
      else:
        for mask_idx, mask in enumerate(masks[bn_list[i-1]]):
          if self.args.use_elbow:
            if self.args.sp_method == 'MBKM':
              clust = cluster.MiniBatchKMeans(batch_size=self.args.mbkm_batch_size)
            else:
              clust = cluster.KMeans()
            # use elbow method to determine the number of clusters
            elbow_alg = kelbow_visualizer(clust,
                                          act.reshape(-1, act.shape[-1]),
                                          k=3,
                                          metric='silhouette',
                                          show=False,
                                          timings=False,
                                          )

            if self.args.cluster_parallel_workers > 0:
              with threadpool_limits(user_api="openmp", limits=self.args.cluster_parallel_workers):
                elbow_alg.fit(act.reshape(-1, act.shape[-1]))
            else:
              elbow_alg.fit(act.reshape(-1, act.shape[-1]))

            n_segments = elbow_alg.elbow_value_
            self.elbow_vals[bn].append(n_segments)
          else:
            # choose n_segments based on ratio of spatial resolution between layeers L-1/L (or at least 2)
            n_segments = np.maximum(int(bn_shapes[bn_list[i]][0]/bn_shapes[bn_list[i-1]][0]), 2)

          if n_segments == None:
            # if no elbow found, then no clusters are found, and keep the same mask as previously
            segments = mask.astype(np.uint8)
          else:
            segments = segmentation.slic(act, mask=mask, n_segments=n_segments, compactness=compactness)

          # filter masks without labels
          for s in range(1, segments.max()+1):
            mask = (segments == s).astype(float)
            # if the mask has any 1 values
            if np.mean(mask) > 0.001:
              masks[bn].append(mask)
              mask_assignment[bn].append(mask_idx)
              mask_labels[bn].append(s-1)

    bn_superpixels = {bn: [] for bn in single_img_acts.keys()}
    bn_patches = {bn: [] for bn in single_img_acts.keys()}
    for bn in single_img_acts.keys():
      for mask in masks[bn]:
        superpixel, patch = self._extract_patch(img, mask, save_bg=save_bg)
        bn_superpixels[bn].append(superpixel)
        bn_patches[bn].append(patch)

    return bn_superpixels, bn_patches, mask_assignment, mask_labels

  def _extract_patch(self, image, mask, save_bg='average'):
    """Extracts a patch out of an image.

    Args:
      image: The original image
      mask: The binary mask of the patch area

    Returns:
      image_resized: The resized patch such that its boundaries touches the
        image boundaries
      patch: The original patch. Rest of the image is padded with average value
    """
    if mask.shape[0] != image.shape[0]:
      mask = resize(mask, output_shape=(image.shape[0], image.shape[1]), order=0).astype(int) # order 0 is nn instead of bilinear
    mask_expanded = np.expand_dims(mask, -1)
    if save_bg == 'white':
        patch = (mask_expanded * image + (1 - mask_expanded))
    else:
        patch = (mask_expanded * image + (1 - mask_expanded) * float(self.average_image_value) / 255)
    ones = np.where(mask == 1)
    h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()

    # dealing with cases where the segments are a single pixel thick
    if h1 == h2:
      h2 += 1
    if w1 == w2:
      w2 += 1

    image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
    image_resized = np.array(image.resize(self.image_shape,Image.Resampling.BICUBIC)).astype(float) / 255
    return image_resized, patch

  def _patch_activations(self, imgs, channel_mean=None, transform=None):
    """Returns activations of a list of imgs.

    Args:
      imgs: List/array of images to calculate the activations of
      bottleneck: Name of the bottleneck layer of the model where activations
        are calculated
      bs: The batch size for calculating activations. (To control computational
        cost)
      channel_mean: If true, the activations are averaged across channel.

    Returns:
      The array of activations
    """
    if channel_mean is None:
      channel_mean = self.channel_mean
    output = {bn: [] for bn in self.bottlenecks}
    for i in range(int(imgs.shape[0] / self.bs) + 1):
      img_batch = imgs[i * self.bs:(i + 1) * self.bs]
      if img_batch.shape[0] == 0:
        # break for no batch
        break
      img_batch_tensor = torchvision.transforms.functional.normalize(
                               torch.tensor(img_batch).cuda().permute(0, 3, 1, 2),
                               mean=self.mean, std=self.std).float()
      if 'clip' in self.args.model_to_run:
        _ = self.model.encode_image(img_batch_tensor)
      else:
        _ = self.model(img_batch_tensor)

      for i, bn in enumerate(self.bottlenecks):
        output[bn].append(features_blobs[0].transpose(0,2,3,1))
        features_blobs.pop(0)
        if self.args.model_to_run.find('vit') > -1 and not self.args.model_to_run == 'mvit':
          # reset cls token collector since we don't need right meow
          cls_token_blobs.pop(0)

    for bn in self.bottlenecks:
      output[bn] = np.concatenate(output[bn], 0)
      if channel_mean and len(output[bn].shape) > 3:
        output[bn] = np.mean(output[bn], (1, 2))
      else:
        output[bn] = np.reshape(output[bn], [output[bn].shape[0], -1])
    return output

  def _cluster(self, acts, method='KM', param_dict=None):
    """Runs unsupervised clustering algorithm on concept actiavtations.

    Args:
      acts: activation vectors of datapoints points in the bottleneck layer.
        E.g. (number of clusters,) for Kmeans
      method: clustering method. We have:
        'KM': Kmeans Clustering
        'MBKM': Minibatch Kmeans Clustering
        'AP': Affinity Propagation
        'SC': Spectral Clustering
        'MS': Mean Shift clustering
        'DB': DBSCAN clustering method
      param_dict: Contains superpixl method's parameters. If an empty dict is
                 given, default parameters are used.

    Returns:
      asg: The cluster assignment label of each data points
      cost: The clustering cost of each data point
      centers: The cluster centers. For methods like Affinity Propagetion
      where they do not return a cluster center or a clustering cost, it
      calculates the medoid as the center  and returns distance to center as
      each data points clustering cost.

    Raises:
      ValueError: if the clustering method is invalid.
    """
    if param_dict is None:
      param_dict = {}
    centers = None
    if method == 'KM':

      if self.args.discovery_n_cluster_method == 'elbow':
        # elbow
        k = 25 if acts.shape[0] > 25 else acts.shape[0]-1
        elbow_alg = kelbow_visualizer(cluster.KMeans(),
                                        acts,
                                        k=k,
                                        metric='silhouette',
                                        show=False,
                                        timings=False,
                                        )
        elbow_alg.fit(acts)
        n_clusters = elbow_alg.elbow_value_
      elif self.args.discovery_n_cluster_method == 'divide':
        n_clusters = int(acts.shape[0]/6)
      else:
        n_clusters = param_dict.pop('n_clusters', 25)
      km = cluster.KMeans(n_clusters)
      d = km.fit(acts)
      centers = km.cluster_centers_
      d = np.linalg.norm(
          np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
      asg, cost = np.argmin(d, -1), np.min(d, -1)
    elif method == 'AP':
      damping = param_dict.pop('damping', 0.5)
      ca = cluster.AffinityPropagation(damping=damping)
      ca.fit(acts)
      centers = ca.cluster_centers_
      d = np.linalg.norm(
          np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
      asg, cost = np.argmin(d, -1), np.min(d, -1)
    elif method == 'MS':
      ms = cluster.MeanShift(n_jobs=1)
      asg = ms.fit_predict(acts)
    elif method == 'SC':
      n_clusters = param_dict.pop('n_clusters', 25)
      sc = cluster.SpectralClustering(n_clusters=n_clusters, n_jobs=1)
      asg = sc.fit_predict(acts)
    elif method == 'DB':
      eps = param_dict.pop('eps', 0.5)
      min_samples = param_dict.pop('min_samples', 20)
      sc = cluster.DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
      asg = sc.fit_predict(acts)
    else:
      raise ValueError('Invalid Clustering Method!')
    if centers is None:  ## If clustering returned cluster centers, use medoids
      centers = np.zeros((asg.max() + 1, acts.shape[1]))
      cost = np.zeros(len(acts))
      for cluster_label in range(asg.max() + 1):
        cluster_idxs = np.where(asg == cluster_label)[0]
        cluster_points = acts[cluster_idxs]
        pw_distances = metrics.euclidean_distances(cluster_points)
        centers[cluster_label] = cluster_points[np.argmin(
            np.sum(pw_distances, -1))]
        cost[cluster_idxs] = np.linalg.norm(
            acts[cluster_idxs] - np.expand_dims(centers[cluster_label], 0),
            ord=2,
            axis=-1)
    return asg, cost, centers

  def discover_concepts(self,
                        method='KM',
                        param_dicts=None):
    """Discovers the frequent occurring concepts in the target class.

      Calculates self.dic, a dictionary containing all the information of the
      discovered concepts in the form of {'bottleneck layer name: bn_dic} where
      bn_dic itself is in the form of {'concepts:list of concepts,
      'concept name': concept_dic} where the concept_dic is in the form of
      {'images': resized patches of concept,
      'patches': original patches of the concepts,
      'image_numbers': image id of each patch}

    Args:
      method: Clustering method.
      activations: If activations are already calculated. If not calculates
                   them. Must be a dictionary in the form of {'bn':array, ...}
      param_dicts: A dictionary in the format of {'bottleneck':param_dict,...}
                   where param_dict contains the clustering method's parametrs
                   in the form of {'param1':value, ...}. For instance for Kmeans
                   {'n_clusters':25}. param_dicts can also be in the format
                   of param_dict where same parameters are used for all
                   bottlenecks.
    """
    if param_dicts is None:
      param_dicts = {}
    self.dic = {}  ## The main dictionary of the ConceptDiscovery class.
    for j, bn in enumerate(self.bottlenecks):
      bn_dic = {}
      # if we segmented based on the features, perform the clustering based only on those features
      img_dir = os.path.join(self.dataset_dir, self.args.seg_type, bn)
      img_list = glob.glob(img_dir + '/*')
      # less memory version
      activations = {layer: [] for layer in self.bottlenecks}
      for i in range(int(len(img_list) / self.bs) + 1):
        img_batch = img_list[i * self.bs:(i + 1) * self.bs]
        imgs = np.array([np.load(x) for x in img_batch])
        if len(imgs) != 0:
          single_activation = self._patch_activations(imgs)
          for layer in self.bottlenecks:
            activations[layer].append(single_activation[layer])
      # concatenate batches
      for layer in self.bottlenecks:
        activations[layer] = np.concatenate(activations[layer],0)

      bn_activations = activations[bn]

      bn_dic['label'], bn_dic['cost'], centers = self._cluster(
          bn_activations, method, param_dicts[bn])
      concept_number, bn_dic['concepts'] = 0, []
      for i in range(bn_dic['label'].max() + 1):
        label_idxs = np.where(bn_dic['label'] == i)[0]

        # filtering method
        if self.args.min_img_method == 'divide':
          curr_min_img = bn_activations.shape[0]/self.min_imgs
        elif self.args.min_img_method == 'og':
          curr_min_img = self.min_imgs
        elif self.args.min_img_method == 'root':
          curr_min_img = bn_activations.shape[0] ** self.args.min_img_root
        elif self.args.min_img_method == 'sigmoid':
          curr_min_img = self.args.min_img_A +((self.args.min_img_K-self.args.min_img_A)/
                                               (self.args.min_img_C+self.args.min_img_Q*
                                                (np.exp(-self.args.min_img_B*bn_activations.shape[0])))**(1/self.args.min_img_v))
        else:
          raise NotImplementedError

        if i == 0:
          print('Min number of images for layer {}: {}'.format(bn, curr_min_img))


        if len(label_idxs) > curr_min_img:
          # filter concepts based on commonality
          concept_costs = bn_dic['cost'][label_idxs]
          concept_idxs = label_idxs[np.argsort(concept_costs)[:self.max_imgs]]
          concept_image_numbers = set(self.bn_image_numbers[bn][label_idxs])
          label_len = len(label_idxs)
          discovery_size = len(self.discovery_images)

          # same as ACE
          highly_common_concept = len(concept_image_numbers) > 0.5 * label_len
          mildly_common_concept = len(concept_image_numbers) > 0.25 * label_len
          non_common_concept = len(concept_image_numbers) > 0.1 * label_len

          # if there are sufficient masks, then we can assume commonality
          if label_len > 4 * self.max_imgs:
            highly_common_concept = True

          # the concept is found in at least quarter/half of discovery images
          mildly_populated_concept = len(concept_image_numbers) > 0.25 * discovery_size
          highly_populated_concept = len(concept_image_numbers) > 0.5 * discovery_size

          cond2 = mildly_populated_concept and mildly_common_concept
          cond3 = non_common_concept and highly_populated_concept

          if highly_common_concept or cond2 or cond3:
            concept_number += 1
            concept = '{}_concept{}'.format(self.target_class, concept_number)
            bn_dic['concepts'].append(concept)
            bn_dic[concept] = {
                'images': [str(x) for x in np.array(self.bn_dataset[bn])[concept_idxs]],
                'patches': [str(x) for x in np.array(self.bn_patches[bn])[concept_idxs]],
                'image_numbers': self.bn_image_numbers[bn][concept_idxs],
                'assignment': self.bn_dataset_mask_assignment[bn][concept_idxs],
            }

            bn_dic[concept + '_center'] = centers[i]
      bn_dic.pop('label', None)
      bn_dic.pop('cost', None)
      self.dic[bn] = bn_dic

  def _random_concept_activations(self, bottleneck, random_concept):
    """Wrapper for computing or loading activations of random concepts.

    Takes care of making, caching (if desired) and loading activations.

    Args:
      bottleneck: The bottleneck layer name
      random_concept: Name of the random concept e.g. "random500_0"

    Returns:
      A nested dict in the form of {concept:{bottleneck:activation}}
    """
    rnd_acts_path = os.path.join(self.activation_dir, 'acts_{}_{}'.format(
        random_concept, bottleneck)) + '.npy'
    if not os.path.exists(rnd_acts_path):
      rnd_imgs = self.load_concept_imgs(random_concept, self.max_imgs, source_dir=self.args.random_dir)
      acts = self.get_acts_from_images(rnd_imgs, bottleneck)
      if self.save_acts:
        np.save(rnd_acts_path, acts, allow_pickle=False)
        del acts
        del rnd_imgs
        return np.load(rnd_acts_path).squeeze()
      else:
        return acts.squeeze()

  def _calculate_cav(self, c, r, bn, act_c, ow, directory=None):
    """Calculates a sinle cav for a concept and a one random counterpart.

    Args:
      c: conept name
      r: random concept name
      bn: the layer name
      act_c: activation matrix of the concept in the 'bn' layer
      ow: overwrite if CAV already exists
      directory: to save the generated CAV

    Returns:
      The accuracy of the CAV
    """
    if directory is None:
      directory = self.cav_dir
    act_r = self._random_concept_activations(bn, r)
    cav_instance = cav.get_or_train_cav([c, r],
                                        bn, {
                                            c: {
                                                bn: act_c
                                            },
                                            r: {
                                                bn: act_r
                                            }
                                        },
                                        cav_dir=directory,
                                        overwrite=ow)
    return cav_instance.accuracies['overall']

  def _concept_cavs(self, bn, concept, activations, randoms=None, ow=True):
    """Calculates CAVs of a concept versus all the random counterparts.

    Args:
      bn: bottleneck layer name
      concept: the concept name
      activations: activations of the concept in the bottleneck layer
      randoms: None if the class random concepts are going to be used
      ow: If true, overwrites the existing CAVs
    Returns:
      A dict of cav accuracies in the form of {'bottleneck layer':
      {'concept name':[list of accuracies], ...}, ...}
    """
    if randoms is None:
      randoms = [
          'random500_{}'.format(i) for i in np.arange(self.num_random_exp)
      ]
    accs = []
    for rnd in randoms:
      accs.append(self._calculate_cav(concept, rnd, bn, activations, ow))
    return accs

  def cavs(self, min_acc=0., ow=True):
    """Calculates cavs for all discovered concepts.

    This method calculates and saves CAVs for all the discovered concepts
    versus all random concepts in all the bottleneck layers

    Args:
      min_acc: Delete discovered concept if the average classification accuracy
        of the CAV is less than min_acc
      ow: If True, overwrites an already calcualted cav.

    Returns:
      A dicationary of classification accuracy of linear boundaries orthogonal
      to cav vectors
    """
    # clear cls_tokens before
    acc = {bn: {} for bn in self.bottlenecks}
    concepts_to_delete = []
    for bn in self.bottlenecks:
      for concept in self.dic[bn]['concepts']:

        concept_imgs = np.array([np.load(x) for x in self.dic[bn][concept][self.args.seg_type]])
        # activations of concept image patches, resized to fill image, at layer bn
        concept_acts = self.get_acts_from_images(concept_imgs, bn)
        # save CAV of concept at layer bn, against random images
        acc[bn][concept] = self._concept_cavs(bn, concept, concept_acts, ow=ow)
        if np.mean(acc[bn][concept]) < min_acc:
          concepts_to_delete.append((bn, concept))
      # activations of full discovery images (e.g., Zebra images) at layer bn
      target_class_acts = self.get_acts_from_images(
          self.discovery_images, bn)
      # save CAV of target class concept at layer bn, against random images
      acc[bn][self.target_class] = self._concept_cavs(
          bn, self.target_class, target_class_acts, ow=ow)
      # random image activations at layer bn
      rnd_acts = self._random_concept_activations(bn, self.random_concept)
      # save and calculate random CAVs (for statistical testing)
      acc[bn][self.random_concept] = self._concept_cavs(
          bn, self.random_concept, rnd_acts, ow=ow)
    for bn, concept in concepts_to_delete:
      self.delete_concept(bn, concept)
    return acc

  def load_cav_direction(self, c, r, bn, directory=None):
    """Loads an already computed cav.
    Args:
      c: concept name
      r: random concept name
      bn: bottleneck layer
      directory: where CAV is saved

    Returns:
      The cav instance
    """
    if directory is None:
      directory = self.cav_dir
    try:
      params = {'model_type':'linear', 'alpha':.01}
      cav_key = cav.CAV.cav_key([c, r], bn, params['model_type'], params['alpha'])
      cav_path = os.path.join(self.cav_dir, cav_key.replace('/', '.') + '.pkl')
      vector = cav.CAV.load_cav(cav_path).cavs[0]
    except:
      params = {'model_type': 'logistic', 'alpha': .01}
      cav_key = cav.CAV.cav_key([c, r], bn, params['model_type'], params['alpha'])
      cav_path = os.path.join(self.cav_dir, cav_key.replace('/', '.') + '.pkl')
      vector = cav.CAV.load_cav(cav_path).cavs[0]
    return np.expand_dims(vector, 0) / np.linalg.norm(vector, ord=2)

  def _sort_concepts(self, scores):
    for bn in self.bottlenecks:
      tcavs = []
      for concept in self.dic[bn]['concepts']:
        tcavs.append(np.mean(scores[bn][concept]))
      concepts = []
      for idx in np.argsort(tcavs)[::-1]:
        concepts.append(self.dic[bn]['concepts'][idx])
      self.dic[bn]['concepts'] = concepts

  def _return_logit_gradients(self, images):
    """For the given images calculates the gradient tensors.

    Args:
      images: Images for which we want to calculate gradients.

    Returns:
      A dictionary of images gradients in all bottleneck layers.
    """
    gradients = {}
    if not self.target_class == 'all':
      try:
        class_id = self.model.label_to_id[self.target_class.replace('_', ' ').replace('-', ' ').lower()]
      except:
        class_id = self.model.label_to_id[self.target_class.lower()]
    for bn in self.bottlenecks:
      acts = self.get_acts_from_images(images,bn, return_cls_token=True)
      if not self.args.model_to_run.find('vit') > -1 or self.args.model_to_run == 'mvit':

        bn_grads = np.zeros((acts.shape[0], np.prod(acts.shape[1:])))
      else:
        # lose class token
        bn_grads = np.zeros((acts.shape[0], np.prod(acts[:,1:,:].shape[1:])))
      # get the gradients of the output class w.r.t the activations
      for i in range(len(acts)):
        if self.target_class == 'all':
          class_id = self.discovery_labels[i]
        grads = self._get_gradients(acts[i:i+1], [class_id], bn, example=None)
        bn_grads[i] = grads.reshape(-1)
      gradients[bn] = bn_grads
    return gradients

  def _get_cutted_model(self, bottleneck, upper_bottleneck=None):
    # get layers only after bottleneck
    new_model_list_keys = []
    new_model_list_vals = []
    add_to_list = False

    # construct layer generator
    if 'vgg' in self.args.model_to_run or self.args.model_to_run == 'alexnet' or self.args.model_to_run == 'resnet18_cub':
      layer_generator = self.model.features.named_children()
    elif self.args.model_to_run == 'clip_r50':
      if '.' in bottleneck:
        layer_generator = {}
        for name, module in self.model.visual.named_modules():
          # only take bottleneck layers
          if len(name.split('.')) < 3 and 'proj' not in name:
            if not isinstance(module, torch.nn.Sequential):
              layer_generator[name] = module
        layer_generator = layer_generator.items()
      else:
        layer_generator = self.model.visual.named_children()
    elif self.args.model_to_run == 'clip_vit':
      layer_generator = self.model._modules['visual']._modules['transformer']._modules['resblocks'].named_children()
    elif self.args.model_to_run.find('vit') > -1:
      layer_generator = self.model.blocks.named_children()
    elif self.args.model_to_run == 'mobilenet_v2' or self.args.model_to_run == 'efficientnet_b4' or self.args.model_to_run == 'mnasnet_a1' or self.args.model_to_run == 'tf_mobilenetv3_large_075':
      if len(bottleneck.split('.')) == 2:
        layer_generator = {}
        for name, module in self.model.blocks.named_modules():
          # only take bottleneck layers
          if len(name.split('.')) < 3:
            if not isinstance(module, torch.nn.Sequential):
              layer_generator[name] = module
        layer_generator = layer_generator.items()
      else:
        layer_generator = self.model.blocks.named_children()
    else:
      if len(bottleneck.split('.')) == 2:
        layer_generator = {}
        for name, module in self.model.named_modules():
          # only take bottleneck layers
          if len(name.split('.')) < 3:
            if not isinstance(module, torch.nn.Sequential):
              if not name == 'global_pool.flatten' and not name == 'global_pool.pool':
                layer_generator[name] = module
        layer_generator = layer_generator.items()
      elif len(bottleneck.split('.')) == 3:
        layer_generator = {}
        for name, module in self.model.named_modules():
          # only take bottleneck layers
          if len(name.split('.')) > 2:
            if not isinstance(module, torch.nn.Sequential):
              layer_generator[name] = module
        layer_generator = layer_generator.items()
      else:
        layer_generator = self.model.named_children()

    # add layers to new model
    for name, layer in layer_generator:
      if add_to_list:
        if not 'aux' in name:
          if name == 'fc':
            new_model_list_keys.append('flatten')
            new_model_list_vals.append(torch.nn.Flatten())
          new_model_list_keys.append(name.replace('.', '-'))
          new_model_list_vals.append(copy.deepcopy(layer))
        if name == upper_bottleneck:
          break
      if name == bottleneck:
        add_to_list = True

    # add final layers to model
    if upper_bottleneck is None:
      if 'vgg' in self.args.model_to_run or self.args.model_to_run == 'tf_mobilenetv3_large_075':
        if self.args.model_to_run == 'tf_mobilenetv3_large_075':
          new_model_list_keys.append('global_pool')
          new_model_list_vals.append(self.model.global_pool)
          new_model_list_keys.append('conv_head')
          new_model_list_vals.append(self.model.conv_head)
          new_model_list_keys.append('act2')
          new_model_list_vals.append(self.model.act2)
          new_model_list_keys.append('flatten')
          new_model_list_vals.append(torch.nn.Flatten())
          new_model_list_keys.append('classifier')
          new_model_list_vals.append(self.model.classifier)
        elif 'vgg' in self.args.model_to_run:
          if 'sin' in self.args.model_to_run:
            new_model_list_keys.append('flatten')
            new_model_list_vals.append(torch.nn.Flatten())
            new_model_list_keys.append('classifier')
            new_model_list_vals.append(self.model.classifier)
          else:
            new_model_list_keys.append('pre_logits')
            new_model_list_vals.append(self.model.pre_logits)
            new_model_list_keys.append('head')
            new_model_list_vals.append(self.model.head)
        elif 'inception_v3' in self.args.model_to_run:
          new_model_list_keys.append('avgpool')
          new_model_list_vals.append(self.model.global_pool)
          new_model_list_keys.append('flatten')
          new_model_list_vals.append(torch.nn.Flatten())
          new_model_list_keys.append('fc')
          new_model_list_vals.append(self.model.fc)
        else:
          new_model_list_keys.append('avgpool')
          new_model_list_vals.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
          new_model_list_keys.append('flatten')
          new_model_list_vals.append(torch.nn.Flatten())
          new_model_list_keys.append('classifier')
          new_model_list_vals.append(self.model.classifier)
      elif 'mvit' in self.args.model_to_run:
        new_model_list_keys.append('norm')
        new_model_list_vals.append(self.model.norm)
        new_model_list_keys.append('head')
        new_model_list_vals.append(self.model.head)
      elif self.args.model_to_run == 'vit_b':
        new_model_list_keys.append('norm')
        new_model_list_vals.append(self.model.fc_norm)
        new_model_list_keys.append('head')
        new_model_list_vals.append(self.model.head)

    new_model_list = OrderedDict()
    for k, v in zip(new_model_list_keys, new_model_list_vals):
      new_model_list[k] = v
    cutted_model = torch.nn.Sequential(new_model_list)

    # remove any existing forward hooks from new model
    for name, layer in cutted_model.named_children():
      if len(layer._forward_hooks.keys()) > 0:
        for key in layer._forward_hooks.keys():
          layer._forward_hooks.pop(key)
    return cutted_model

  def _get_gradients(self, acts, y, bottleneck_name, example=None):
    # TCAV method gradients
    inputs = torch.autograd.Variable(torch.tensor(acts).cuda(), requires_grad=True)
    cutted_model = self._get_cutted_model(bottleneck_name).cuda()
    cutted_model.eval()
    cutted_model.zero_grad()

    if 'mvit' in self.args.model_to_run:
      inputs = inputs.reshape(1, -1, inputs.shape[-1])
      outputs = cutted_model(inputs).type(torch.float32)
    elif 'clip' in self.args.model_to_run:
      if 'r50' in self.args.model_to_run:
        image_features = cutted_model(inputs.permute(0,3,1,2)).type(torch.float32)
      elif 'vit' in self.args.model_to_run:
        image_features = cutted_model(inputs.permute(1,0,2))
        image_features = image_features.permute(1,0,2)
        image_features = self.model.visual.ln_post(image_features[:, 0, :])
        image_features = image_features @ self.model.visual.proj

      # normalized features
      image_features = (image_features / image_features.norm(dim=-1, keepdim=True)).type(torch.float32)

      # cosine similarity as logits
      logit_scale = self.model.logit_scale.exp().type(torch.float32)
      outputs = logit_scale * image_features @ self.model.text_features.t()

    elif self.args.model_to_run == 'vit_b':
      outputs = cutted_model(inputs).type(torch.float32)[:,0]
    else:
      inputs = inputs.permute(0,3,1,2)
      outputs = cutted_model(inputs).type(torch.float32)

    grads = torch.autograd.grad(outputs[:, y[0]], inputs)[0]

    if self.args.model_to_run.find('vit') > -1 and not self.args.model_to_run == 'mvit':
      grads = grads[:,1:,:]
    grads = grads.detach().cpu().numpy()

    cutted_model = None
    gc.collect()
    # print(outputs.argmax().item(), y[0])
    return grads


  def _return_inter_concept_gradients(self, images, lower_bn, upper_bn, upper_cluster_center):
    # get activations at lower_bn of upper concept patches
    acts = self.get_acts_from_images(images,lower_bn, return_cls_token=True)
    if not self.args.model_to_run.find('vit') > -1 or self.args.model_to_run == 'mvit':
      bn_grads = np.zeros((acts.shape[0], np.prod(acts.shape[1:])))
    else:
      # gotta lose class token
      bn_grads = np.zeros((acts.shape[0], np.prod(acts[:,1:,:].shape[1:])))


    # get the gradients of the output class w.r.t the activations
    # get the gradients of the upper concept cluster center w.r.t the activations
    for i in range(len(acts)):
      grads = self._get_inter_gradients(acts[i:i+1], upper_cluster_center, lower_bn, upper_bn, example=None)
      bn_grads[i] = grads.reshape(-1)
    gradients = bn_grads
    return gradients

  def _get_inter_gradients(self, acts, cluster_center, lower_bn, upper_bn, example=None):
    # ITCAV method gradients
    inputs = torch.autograd.Variable(torch.tensor(acts).cuda(), requires_grad=True)
    cluster_center = torch.tensor(cluster_center).cuda()
    # get model from layer lower_bn to upper_bn
    cutted_model = self._get_cutted_model(lower_bn, upper_bn).cuda()
    cutted_model.eval()

    if 'mvit' in self.args.model_to_run:
      inputs = inputs.reshape(1, -1, inputs.shape[-1])
      outputs = cutted_model(inputs).type(torch.float32)
    elif self.args.model_to_run == 'vit_b':
      outputs = cutted_model(inputs).type(torch.float32)[:,1:]
    elif self.args.model_to_run == 'clip_vit':
      outputs = cutted_model(inputs).type(torch.float32)[:,1:]
    else:
      outputs = cutted_model(inputs.permute(0,3,1,2)).type(torch.float32)

    # measure alignment scoreof output with concept cluster center, take mean of output if needed
    if self.channel_mean:
      if self.args.model_to_run.find('vit') > -1:
        outputs = outputs.mean((0,1))
      else:
        outputs = outputs.mean((0,2,3))
    else:
      outputs = outputs.reshape(-1)

    # calculate the distance between the output and the cluster center
    outputs = ((outputs - cluster_center) ** 2).sum().sqrt()

    # take gradient of score w.r.t input activations
    grads = -torch.autograd.grad(outputs=outputs, inputs=inputs)[0]

    # if vit, remove class token
    if self.args.model_to_run.find('vit') > -1 and not self.args.model_to_run == 'mvit':
      grads = grads[:,1:,:]
    grads = grads.detach().cpu().numpy()
    gc.collect()
    return grads


  def _tcav_score(self, bn, concept, rnd, gradients):
    """Calculates and returns the TCAV score of a concept.

    Args:
      bn: bottleneck layer
      concept: concept name
      rnd: random counterpart
      gradients: Dict of gradients of tcav_score_images
    Returns:
      TCAV score of the concept with respect to the given random counterpart
    """
    # load CAV trained on concept and rnd, for layer bn
    vector = self.load_cav_direction(concept, rnd, bn)
    # for the activations of the target images at layer bn (from tcav_score_images), evaluate the alignment with the CAV and the CAV
    prod = np.sum(gradients[bn] * vector, -1)
    # count the number of positive alignments
    num_positive = np.mean(prod > 0)
    return num_positive

  def target_class_tcavs(self, test=False, sort=True, tcav_score_images=None):
    """Calculates TCAV scores for all discovered concepts and sorts concepts.

    This method calculates TCAV scores of all the discovered concepts for
    the target class using all the calculated cavs. It later sorts concepts
    based on their TCAV scores.

    Args:
      test: If true, perform statistical testing and removes concepts that don't
        pass
      sort: If true, it will sort concepts in each bottleneck layers based on
        average TCAV score of the concept.
      tcav_score_images: Target class images used for calculating tcav scores.
        If None, the target class source directory images are used.

    Returns:
      A dictionary of the form {'bottleneck layer':{'concept name':
      [list of tcav scores], ...}, ...} containing TCAV scores.
    """
    tcav_scores = {bn: {} for bn in self.bottlenecks}
    randoms = ['random500_{}'.format(i) for i in np.arange(self.num_random_exp)]
    if tcav_score_images is None:  # Load target class images if not given
      if 'toy' in self.args.target_dataset:
        tcav_score_images = self.discovery_images
      else:
        raw_imgs = self.load_concept_imgs(self.target_class, 2 * self.args.max_imgs)
        tcav_score_images = raw_imgs[-self.args.max_imgs:]
    # get gradients of the target logit w.r.t each bn layer, then flatten, of the target class discovery images
    gradients = self._return_logit_gradients(tcav_score_images)
    for bn in self.bottlenecks:
      for concept in self.dic[bn]['concepts'] + [self.random_concept]:
        def t_func(rnd):
          tcav_score = self._tcav_score(bn, concept, rnd, gradients)
          return tcav_score
        # calculate the TCAV score between concept at layer bn, and all other random concpets
        tcav_scores[bn][concept] = [t_func(rnd) for rnd in randoms]
    if test:
      self.test_and_remove_concepts(tcav_scores)
    if sort:
      self._sort_concepts(tcav_scores)
    return tcav_scores

  def do_statistical_testings(self, i_ups_concept, i_ups_random):
    """Conducts ttest to compare two set of samples.

    In particular, if the means of the two samples are staistically different.

    Args:
      i_ups_concept: samples of TCAV scores for concept vs. randoms
      i_ups_random: samples of TCAV scores for random vs. randoms

    Returns:
      p value
    """
    min_len = min(len(i_ups_concept), len(i_ups_random))
    _, p = stats.ttest_rel(i_ups_concept[:min_len], i_ups_random[:min_len])
    return p

  def test_and_remove_concepts(self, tcav_scores):
    """Performs statistical testing for all discovered concepts.

    Using TCAV scores of the discovered concepts versus the random_counterpart
    concept, performs statistical testing and removes concepts that do not pass

    Args:
      tcav_scores: Calculated dicationary of tcav scores of all concepts
    """
    concepts_to_delete = []
    for bn in self.bottlenecks:
      for concept in self.dic[bn]['concepts']:
        pvalue = self.do_statistical_testings(tcav_scores[bn][concept], tcav_scores[bn][self.random_concept])
        if pvalue > 0.05:
          concepts_to_delete.append((bn, concept))
    for bn, concept in concepts_to_delete:
      self.delete_concept(bn, concept)

  def delete_concept(self, bn, concept):
    """Removes a discovered concepts if it's not already removed.

    Args:
      bn: Bottleneck layer where the concepts is discovered.
      concept: concept name
    """
    self.dic[bn].pop(concept, None)
    if concept in self.dic[bn]['concepts']:
      self.dic[bn]['concepts'].pop(self.dic[bn]['concepts'].index(concept))

  def _concept_profile(self, bn, activations, concept, randoms):
    """Transforms data points from activations space to concept space.

    Calculates concept profile of data points in the desired bottleneck
    layer's activation space for one of the concepts

    Args:
      bn: Bottleneck layer
      activations: activations of the data points in the bottleneck layer
      concept: concept name
      randoms: random concepts

    Returns:
      The projection of activations of all images on all CAV directions of
        the given concept
    """
    def t_func(rnd):
      products = self.load_cav_direction(concept, rnd, bn) * activations
      return np.sum(products, -1)
    profiles = [t_func(rnd) for rnd in randoms]
    return np.stack(profiles, axis=-1)

  def find_profile(self, bn, images, mean=True):
    """Transforms images from pixel space to concept space.

    Args:
      bn: Bottleneck layer
      images: Data points to be transformed
      mean: If true, the profile of each concept would be the average inner
        product of all that concepts' CAV vectors rather than the stacked up
        version.

    Returns:
      The concept profile of input images in the bn layer.
    """
    profile = np.zeros((len(images), len(self.dic[bn]['concepts']),
                        self.num_random_exp))
    class_acts = self.get_acts_from_images(
        images).reshape([len(images), -1])
    randoms = ['random500_{}'.format(i) for i in range(self.num_random_exp)]
    for i, concept in enumerate(self.dic[bn]['concepts']):
      profile[:, i, :] = self._concept_profile(bn, class_acts, concept, randoms)
    if mean:
      profile = np.mean(profile, -1)
    return profile



  def get_acts_from_images(self, imgs, bottleneck=None, return_cls_token=False):
    """Run images in the model to get the activations.
    Args:
      imgs: a list of images
      model: a model instance
      bottleneck_name: bottleneck name to get the activation from
    Returns:
      numpy array of activations.
    """
    output = {bn: [] for bn in self.bottlenecks}
    if 'clip' in self.args.model_to_run:
      # if using clip model, need to save GPU space...
      for i in range(imgs.shape[0]):
        img_batch_tensor = torchvision.transforms.functional.normalize(
          torch.tensor(imgs[i]).unsqueeze(0).cuda().permute(0, 3, 1, 2),
          mean=self.mean, std=self.std).float()

        _ = self.model.encode_image(img_batch_tensor)

        if 'r50' in self.args.model_to_run:
          for i, bn in enumerate(self.bottlenecks):
            output[bn].append(features_blobs[0].transpose(0, 2, 3, 1))
            features_blobs.pop(0)
        elif 'vit' in self.args.model_to_run:
          for i, bn in enumerate(self.bottlenecks):
            if not return_cls_token:
              output[bn].append(features_blobs[0].transpose(0, 2, 3, 1))
              features_blobs.pop(0)
              # reset cls token collector since we don't need right meow
              cls_token_blobs.pop(0)
            else:
              # combine features and cls token
              feature = features_blobs[0]
              cls_token = cls_token_blobs[0]
              cls_feat = np.concatenate([np.expand_dims(cls_token, 1),
                                         feature.reshape(feature.shape[0], feature.shape[1], -1).transpose(0, 2, 1)], 1)
              output[bn].append(cls_feat)
              features_blobs.pop(0)
              cls_token_blobs.pop(0)

      for bn in self.bottlenecks:
        output[bn] = [np.concatenate(output[bn])]
    elif self.args.model_to_run.find('vit') > -1 and not self.args.model_to_run == 'mvit':
      img_batch_tensor = torchvision.transforms.functional.normalize(
        torch.tensor(imgs).cuda().permute(0, 3, 1, 2),
        mean=self.mean, std=self.std).float()
      _ = self.model(img_batch_tensor)

      for i, bn in enumerate(self.bottlenecks):
        if not return_cls_token:
          output[bn].append(features_blobs[0].transpose(0, 2, 3, 1))
          features_blobs.pop(0)
            # reset cls token collector since we don't need right meow
          cls_token_blobs.pop(0)
        else:
          # combine features and cls token
          feature = features_blobs[0]
          cls_token = cls_token_blobs[0]
          cls_feat = np.concatenate([np.expand_dims(cls_token, 1), feature.reshape(feature.shape[0],feature.shape[1],-1).transpose(0,2,1)], 1)
          output[bn].append(cls_feat)
          features_blobs.pop(0)
          cls_token_blobs.pop(0)
    else:
      img_batch_tensor = torchvision.transforms.functional.normalize(
        torch.tensor(imgs).cuda().permute(0, 3, 1, 2),
        mean=self.mean, std=self.std).float()
      _ = self.model(img_batch_tensor)

      for i, bn in enumerate(self.bottlenecks):
        output[bn].append(features_blobs[0].transpose(0, 2, 3, 1))
        features_blobs.pop(0)

    if bottleneck is None:
      return output[0]
    else:
      return output[bottleneck][0]



# hooks for all models
cls_token_blobs = []
features_blobs = []
def hook_feature(module, input, output):
  # if vit, grab tensor and reshape to b x h x w x c
  if output.shape[0] == 50 and len(output.shape) == 3:
    output = output.permute(1, 0, 2)
  if len(output.shape) == 3:
    size = int(np.sqrt(output.shape[1]))
    try:
      output = output.reshape(output.shape[0], size, size, -1).permute(0,3,1,2)
    except:
      cls_token_blobs.append(output[:,0,:].data.cpu().numpy())
      output = output[:,1:,:].reshape(output.shape[0], size, size, -1).permute(0, 3, 1, 2)

  features_blobs.append(output.data.cpu().numpy())

def hook_feature_mvit(module, input, output):
  if len(output.shape) == 3:
    size = int(np.sqrt(output.shape[1]))
    try:
      output = output.reshape(output.shape[0], size, size, -1).permute(0,3,1,2)
    except:
      cls_token_blobs.append(output[:,0,:].data.cpu().numpy())
      output = output[:,1:,:].reshape(output.shape[0], size, size, -1).permute(0, 3, 1, 2)

  features_blobs.append(output.data.cpu().numpy())

def make_model(settings, hook=True):
  import timm
  torch.hub.set_dir('./')
  if not settings.pretrained:
    settings.pretrained = None

      # load model
  if settings.model_to_run == 'clip_r50':
    model, preprocess = clip.load("RN50", device="cuda")
  elif settings.model_to_run == 'mvit':
    from models.mvit.tools.main import parse_args
    from models.mvit.tools.main import load_config
    from models.mvit.mvit.config.defaults import assert_and_infer_cfg
    from models.mvit.mvit.models.build import build_model
    import models.mvit.mvit.utils.checkpoint as cu

    model_args = parse_args()
    model_args.cfg_file = 'models/mvit/configs/MVITv2_S.yaml'
    cfg = load_config(model_args)
    cfg = assert_and_infer_cfg(cfg)
    cfg.NUM_GPUS = 1
    model = build_model(cfg)
    cfg.TRAIN.CHECKPOINT_FILE_PATH = 'checkpoints/MViTv2_S_in1k.pyth'
    cfg.TEST.CHECKPOINT_FILE_PATH = 'checkpoints/MViTv2_S_in1k.pyth'
    cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
    cu.load_test_checkpoint(cfg, model)

  # timm imports
  elif settings.model_to_run == 'resnet18':
    model = timm.create_model('resnet18', pretrained=settings.pretrained)
  elif settings.model_to_run == 'resnet50':
    model = timm.create_model('resnet50', pretrained=settings.pretrained)
  elif settings.model_to_run == 'vgg11':
    model = timm.create_model('vgg11', pretrained=settings.pretrained)
  elif settings.model_to_run == 'vgg16':
    model = timm.create_model('vgg16', pretrained=settings.pretrained)
  elif settings.model_to_run == 'mobilenet_v2':
    model = timm.create_model('mobilenetv2_050', pretrained=settings.pretrained)
  elif settings.model_to_run == 'efficientnet_b4':
     model = timm.create_model('efficientnet_b4', pretrained=settings.pretrained)
  elif settings.model_to_run == 'mnasnet_a1':
    model = timm.create_model('mnasnet_a1', pretrained=settings.pretrained)
  elif settings.model_to_run == 'vit_b':
    model = timm.create_model('vit_base_patch32_224', pretrained=settings.pretrained)
  elif settings.model_to_run == 'inception_v3':
    model = timm.create_model('inception_v3', pretrained=settings.pretrained)
  elif settings.model_to_run == 'tf_mobilenetv3_large_075':
    model = timm.create_model('tf_mobilenetv3_large_075', pretrained=settings.pretrained)

  elif settings.model_to_run == 'resnet50_sin':
    # https://github.com/rgeirhos/texture-vs-shape/blob/master/models/load_pretrained_models.py
    sin_model_urls = {
            'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    }
    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    from torch.utils import model_zoo
    checkpoint = model_zoo.load_url(sin_model_urls['resnet50_trained_on_SIN'])
    model.load_state_dict(checkpoint["state_dict"])
    # remove model from DataParallel wrapper
    model = model.module
  elif settings.model_to_run == 'vgg16_sin':
    # download model from URL manually and save to desired location
    filepath = "checkpoints/vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar"

    assert os.path.exists(
      filepath), "Please download the VGG model yourself from the following link and save it locally: https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK (too large to be downloaded automatically like the other models)"

    model = torchvision.models.vgg16(pretrained=False)
    model.features = torch.nn.DataParallel(model.features)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["state_dict"])
    model.features = model.features.module
  else:
    checkpoint = torch.load(settings.model_path)
    if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
      model = torchvision.models.__dict__[settings.model_to_run](num_classes=settings.target_num_classes)
      if 'state_dict' in list(checkpoint.keys()):
        print('Epoch: {}  -  Acc1: {}'.format(checkpoint['epoch'],checkpoint['acc1']))
        checkpoint = checkpoint['state_dict']
        exit()
      for key, value in checkpoint.items():
        if 'module' in key:
          state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
            'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
        else:
          state_dict = checkpoint
        break
      print('Loading target model checkpoint from: {}'.format(settings.model_path))
      model.load_state_dict(state_dict, strict=True)
    else:
      model = checkpoint
  if hook:
    for name in settings.feature_names:
      if settings.model_to_run == 'alexnet':
        model._modules['features']._modules.get(name).register_forward_hook(hook_feature)
      elif settings.model_to_run == 'mobilenet_v2' or settings.model_to_run == 'efficientnet_b4' or settings.model_to_run == 'mnasnet_a1' or settings.model_to_run == 'tf_mobilenetv3_large_075':
        # model._modules['features']._modules.get(name).register_forward_hook(hook_feature) # toch hub
        if '.' in name:
          module_name = name.split('.')[0]
          layer_name = name.split('.')[1]
          model._modules['blocks']._modules.get(module_name)._modules.get(layer_name).register_forward_hook(hook_feature)
        else:
          model._modules['blocks']._modules.get(name).register_forward_hook(hook_feature) # timm
      elif 'vgg' in settings.model_to_run:
        model._modules['features']._modules.get(name).register_forward_hook(hook_feature)
      elif settings.model_to_run == 'clip_r50':
        if '.' in name:
          module_name = name.split('.')[0]
          layer_name = name.split('.')[1]
          # model._modules.get(module_name)._modules.get(layer_name).register_forward_hook(hook_feature)
          model._modules['visual']._modules.get(module_name)._modules.get(layer_name).register_forward_hook(hook_feature)
        else:
          model._modules['visual']._modules.get(name).register_forward_hook(hook_feature)
      elif settings.model_to_run == 'clip_vit':
        model._modules['visual']._modules['transformer']._modules['resblocks']._modules.get(name).register_forward_hook(hook_feature)
      elif settings.model_to_run == 'mvit':
        model._modules['blocks']._modules.get(name).register_forward_hook(hook_feature_mvit)
      elif settings.model_to_run.find('vit') > -1:
        model._modules['blocks']._modules.get(name).register_forward_hook(hook_feature)
      elif settings.model_to_run.find('cub') > -1:
        model._modules['features']._modules.get(name).register_forward_hook(hook_feature)

      # for name, module in self.model.named_modules():
      else:
        if '.' in name:
          if len(name.split('.')) == 2:
            module_name = name.split('.')[0]
            layer_name = name.split('.')[1]
            model._modules.get(module_name)._modules.get(layer_name).register_forward_hook(hook_feature)
          elif len(name.split('.')) == 3:
            model._modules.get(name.split('.')[0])._modules.get(name.split('.')[1])._modules.get(name.split('.')[2]).register_forward_hook(hook_feature)
        else:
          model._modules.get(name).register_forward_hook(hook_feature)
    model.cuda()
  else:
    model.cuda()
  model.eval()
  if settings.target_dataset == 'toy1':
    model.label_to_id = {'00': 0,'10': 1,'20': 2,'01': 3,'11': 4,'21': 5,'02': 6,'12': 7,'22': 8}
  elif settings.target_dataset == 'toy2' or settings.target_dataset == 'toy3':
    model.label_to_id = {
            '0000': 0,
            '1000': 1,
            '0100': 2,
            '0010': 3,
            '0001': 4,
            '1100': 5,
            '1010': 6,
            '1001': 7,
            '0110': 8,
            '0101': 9,
            '0011': 10,
            '1110': 11,
            '1101': 12,
            '1011': 13,
            '0111': 14,
            '1111': 15,
        }
  elif settings.target_dataset == 'cub':
    classes = [cls.split('/')[-1].lower() for cls in glob.glob(settings.cub_path + '/*')]
    model.labels = [x.split('.')[1].lower() for x in classes]
    model.label_to_id = {v: k for (k, v) in enumerate(model.labels)}
    model.class_idx = {k: v for (k, v) in enumerate(model.labels)}
  else:
    with open(settings.labels_path, 'r') as f:
      class_idx = json.load(f)
    label_list = [class_idx[str(k)][1].replace('-', ' ').replace('_', ' ').lower() for k in range(len(class_idx))]
    model.labels = label_list
    model.label_to_id = {v: k for (k, v) in enumerate(model.labels)}
    model.class_idx = class_idx

  if 'clip' in settings.model_to_run:
    token_label_list = [f"a photo of a {c}" for c in label_list]
    text = clip.tokenize(token_label_list).cuda()
    model.text_features = model.encode_text(text)
    model.text_features = (model.text_features / model.text_features.norm(dim=-1, keepdim=True)).type(torch.float32)

  return model