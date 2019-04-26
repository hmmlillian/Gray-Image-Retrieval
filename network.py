from config import cfg
import numpy as np
import cv2
from blob import im_list_to_blob

def get_image_blob(im):
	"""Converts an image into a network input.

	Arguments:
		im (ndarray): a color image in BGR order

	Returns:
		blob (ndarray): a data blob holding an image pyramid
		im_scale_factors (list): list of image scales (relative to im) used
			in the image pyramid
	"""
	im_orig = im.astype(np.float32, copy=True)
	im_orig -= cfg.PIXEL_MEANS

	im_shape = im_orig.shape
	im_size_min = np.min(im_shape[0:2])
	im_size_max = np.max(im_shape[0:2]) # HMM: scale short edge rather than long edge

	processed_ims = []
	im_scale_factors = []

	im_scale = float(cfg.TEST_MIN_SIZE) / float(im_size_min)
	# Prevent the biggest axis from being more than MAX_SIZE
	if np.round(im_scale * im_size_max) > cfg.TEST_MAX_SIZE:
		im_scale = float(cfg.TEST_MAX_SIZE) / float(im_size_max)

	im = cv2.resize(im_orig, None, None,
					fx = im_scale, fy = im_scale,
					interpolation = cv2.INTER_LINEAR)

	im_scale_factors.append(im_scale)
	processed_ims.append(im)

	# Create a blob to hold the input images
	blob = im_list_to_blob(processed_ims)

	return blob, np.array(im_scale_factors), processed_ims[0].shape


def get_blobs_ftr(im, rois):
	"""Convert an image and RoIs within that image into network inputs."""
	blobs = {'data': None, 'rois': None}
	blobs['data'], im_scale_factors, im_sizes = get_image_blob(im)

	roi_num = len(rois)
	roi_data = np.zeros((roi_num, 5), dtype=np.float32)

	for i in range(roi_num):
		roi_data[i] = [0, rois[i][0], rois[i][1], rois[i][2], rois[i][3]]
	blobs['rois'] = roi_data

	return blobs, im_scale_factors, im_sizes


def extract_ftr(net, im, boxes=None):
	"""Extract features (fc and pool from pretrained network.

	Arguments:
		net (caffe.Net): Pretrained network to use
		im (ndarray): color image to test (in BGR order)
		boxes (ndarray): 1 x 4 array of image shape or None (for RPN)

	Returns:
		fc_vec (ndarray): 1 x 4096 array of fc features
		poolvec (ndarray): 512 x 7 x 7 array of pool features
	"""

	blobs, im_scales, im_sizes = get_blobs_ftr(im, boxes)

	# reshape network inputs
	net.blobs['data'].reshape(*(blobs['data'].shape))
	net.blobs['rois'].reshape(*(blobs['rois'].shape))

	# do forward
	forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
	forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
	blobs_out = net.forward(**forward_kwargs)

	# read features from network
	fc_features = net.blobs[cfg.FC_LAYER].data
	pool_features = net.blobs[cfg.POOL_LAYER].data

	fc_vec = fc_features.astype(np.float32, copy=False)
	pool_vec = pool_features.astype(np.float32, copy=False)

	return fc_vec, pool_vec
