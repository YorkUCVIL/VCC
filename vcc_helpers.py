""" collection of various helper functions for running VCC"""
import os
import pickle
import numpy as np
from PIL import Image

def calc_edge_from_inter_tcav(cd, tcav_scores, class_name):
    edge_weights = {bn: {} for bn in cd.bottlenecks}
    randoms = ['random500_{}'.format(i) for i in np.arange(cd.num_random_exp)]

    num_cc_layers = len(cd.bottlenecks)
    for layer_num in range(num_cc_layers):
        lower_bn = cd.bottlenecks[layer_num]

        # check if we are at the last layer
        if lower_bn == cd.bottlenecks[-1]:
            # if we are, we use tcav scores
            for concept in cd.dic[lower_bn]['concepts'] + [cd.random_concept]:
                # get average TCAV score over random statistical tests as final score
                concept_edge_weight = tcav_scores[lower_bn][concept]
                edge_weights[lower_bn]['class' + ' ' + class_name + '-' + lower_bn + ' ' + concept] = concept_edge_weight
            # we are done after the last layer
            break
        upper_bn = cd.bottlenecks[layer_num+1]
        for lower_concept in cd.dic[lower_bn]['concepts'] + [cd.random_concept]:
            for upper_concept in cd.dic[upper_bn]['concepts']:
                # grab resized patches from upper concept for itcav calculations, we only need CAVs for lower concepts
                upper_concept_dataset = np.array([np.load(x) for x in cd.dic[upper_bn][upper_concept][cd.args.seg_type]])
                upper_concept_cluster_center = cd.dic[upper_bn][upper_concept + '_center']
                # get gradients of upper_concept w.r.t lower_bn activations of upper concept images
                upper_grad_wrt_lower = cd._return_inter_concept_gradients(upper_concept_dataset,
                                                               lower_bn,
                                                               upper_bn,
                                                               upper_concept_cluster_center)

                gradients = {lower_bn: upper_grad_wrt_lower}
                itcav_scores = []
                for rnd in randoms:
                    score = cd._tcav_score(lower_bn, lower_concept, rnd, gradients)
                    itcav_scores.append(score)

                # add to edge weights
                edge_weights[lower_bn][upper_bn + ' ' + upper_concept + '-' + lower_bn + ' ' + lower_concept] = itcav_scores
    return edge_weights


def load_image_from_file(filename, shape):
  """Given a filename, try to open the file. If failed, return None.
  Args:
    filename: location of the image file
    shape: the shape of the image file to be scaled
  Returns:
    the image if succeeds, None if fails.
  Rasies:
    exception if the image was not the right shape.
  """

  if not os.path.exists(filename):
    print('Cannot find file: {}'.format(filename))
    return None
  try:
    img = np.array(Image.open(filename).resize(shape, Image.BILINEAR))
    img = np.float32(img) / 255.0

    if not (len(img.shape) == 3 and img.shape[2] == 3):
      return None
    else:
      return img

  except Exception as e:
    print(e)
    return None
  return img

def load_images_from_files(filenames, max_imgs=500, return_filenames=False,
                           do_shuffle=True,
                           shape=(299, 299)):
  """Return image arrays from filenames.
  Args:
    filenames: locations of image files.
    max_imgs: maximum number of images from filenames.
    return_filenames: return the succeeded filenames or not
    do_shuffle: before getting max_imgs files, shuffle the names or not
    shape: desired shape of the image
    num_workers: number of workers in parallelization.
  Returns:
    image arrays and succeeded filenames if return_filenames=True.
  """
  imgs = []
  # First shuffle a copy of the filenames.
  filenames = filenames[:]
  if do_shuffle:
    np.random.shuffle(filenames)
  if return_filenames:
    final_filenames = []
  for filename in filenames:
    img = load_image_from_file(filename, shape)
    if img is not None:
      imgs.append(img)
      if return_filenames:
        final_filenames.append(filename)
    if len(imgs) >= max_imgs:
      break

  if return_filenames:
    return np.array(imgs), final_filenames
  else:
    return np.array(imgs)

def save_ace_report(cd, accs, scores, address):
  """Saves TCAV scores.

  Saves the average CAV accuracies and average TCAV scores of the concepts
  discovered in ConceptDiscovery instance.

  Args:
    cd: The ConceptDiscovery instance.
    accs: The cav accuracy dictionary returned by cavs method of the
      ConceptDiscovery instance
    scores: The tcav score dictionary returned by tcavs method of the
      ConceptDiscovery instance
    address: The address to save the text file in.
  """
  report = '\n\n\t\t\t ---CAV accuracies---'
  for bn in cd.bottlenecks:
    report += '\n'
    for concept in cd.dic[bn]['concepts']:
      report += '\n' + bn + ':' + concept + ':' + str(
          np.mean(accs[bn][concept]))
  with open(address + 'CAV_ace_results.txt', 'w') as f:
    f.write(report)
  report = '\n\n\t\t\t ---TCAV scores---'
  for bn in cd.bottlenecks:
    report += '\n'
    for concept in cd.dic[bn]['concepts']:
      pvalue = cd.do_statistical_testings(
          scores[bn][concept], scores[bn][cd.random_concept])
      report += '\n{}:{}:{},{}'.format(bn, concept,
                                       np.mean(scores[bn][concept]), pvalue)
  with open(address + 'TCAV_ace_results.txt', 'w') as f:
    f.write(report)

def save_images(addresses, images):
  """Save images in the addresses.

  Args:
    addresses: The list of addresses to save the images as or the address of the
      directory to save all images in. (list or str)
    images: The list of all images in numpy uint8 format.
  """
  if not isinstance(addresses, list):
    image_addresses = []
    for i, image in enumerate(images):
      image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
      image_addresses.append(os.path.join(addresses, image_name))
    addresses = image_addresses
  assert len(addresses) == len(images), 'Invalid number of addresses'
  for address, image in zip(addresses, images):
    new_p = Image.fromarray(image)
    if new_p.mode != 'RGB':
      new_p = new_p.convert('RGB')
    new_p.save(address, format='PNG')

def clean_and_save_cd(args, cd, save_dir, dataset_dir):

  # replace paths with images
  for bn in cd.dic.keys():
    print('Number of concepts at layer {}: {}'.format(bn, len(cd.dic[bn]['concepts'])))
    for concept in cd.dic[bn]['concepts']:
      img_tmp = []
      patch_tmp = []
      for i in range(len(cd.dic[bn][concept]['images'])):
        img_tmp.append(np.load(cd.dic[bn][concept]['images'][i]))
        patch_tmp.append(np.load(cd.dic[bn][concept]['patches'][i]))
      # replace
      cd.dic[bn][concept]['images'] = np.stack(img_tmp)
      cd.dic[bn][concept]['patches'] = np.stack(patch_tmp)

  # clean dict before saving
  cd.model_name = cd.model.__class__.__name__
  delattr(cd, 'model')
  with open(save_dir + '/cd.pkl', 'wb') as f:
    pickle.dump(cd, f)




