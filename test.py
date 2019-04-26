import _init_paths

import numpy as np
import caffe, cv2, os
from ctypes import *
import argparse
from network import extract_ftr
from config import cfg

def get_args():
    parser = argparse.ArgumentParser("""GrayImageRetrieval: Grayscale Image Retrieval based on Deep Features""")

    parser.add_argument("--in_path",  type = str, default = "demo/test/imgs/",
                        help = "the input root folder of test grayscale images")
    parser.add_argument("--out_path", type = str, default = "demo/test/pairs/",
                        help = "the output root folder of text files with search results inside")
    parser.add_argument("--model_path", type=str, default='model/vgg_19_gray_bn/',
                        help="the folder of pretrained VGG model on grayscale images with its deploy files")
    parser.add_argument("--imagenet_path", type=str, default='demo/ImageNet/',
                        help="the folder of compreseed features of ImageNet images")
    parser.add_argument("--gpu_id", type = int, default = 0,
                        help = "ID of the GPU used to forward network")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()

    # create output folder if it does not exist
    if os.path.isdir(opt.out_path) is False:
        os.mkdir(opt.out_path)

    # load caffe model
    prototxt = os.path.join(opt.model_path, 'deploy.prototxt')
    prototxt_roi = os.path.join(opt.model_path, 'deploy_roi.prototxt')
    caffemodel = os.path.join(opt.model_path, 'vgg19_bn_gray_ft_iter_150000.caffemodel')

    caffe.set_mode_gpu()
    caffe.set_device(opt.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net_roi = caffe.Net(prototxt_roi, caffemodel, caffe.TEST)

    # load ImageNet class names
    imagenet_class = os.path.join('model/', 'imagenet_class.txt')
    imagenet_file = open(imagenet_class, 'r')
    cls_names = imagenet_file.read().split('\n')
    imagenet_file.close()
    big_files = ['%d.big' % i for i in range(1000)]
    real_big_class = sorted(range(1000), key = lambda k: big_files[k])

    # get functions in Matlab and C++
    pca_func = _init_paths.get_pca_func()
    search_func = _init_paths.get_search_func()
    #######################################################################################

    # search similar images in ImageNet for each grayscale input image
    img_list = [f for f in os.listdir(opt.in_path) if os.path.isfile(os.path.join(opt.in_path, f))]
    for i in img_list:
        img_name = os.path.join(opt.in_path, i)

        # load and resize image
        orig_img = cv2.imread(img_name)
        min_len = min(orig_img.shape[0], orig_img.shape[1])
        scalar = cfg.TEST_MIN_SIZE / min_len
        img = cv2.resize(orig_img, (int(scalar * orig_img.shape[1]), int(scalar * orig_img.shape[0])))

        # convert to grayscale image
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        img = cv2.merge((l, l, l))
        #######################################################################################


        # Step 1: classify image and find its class name in ImageNet
        data_img = cv2.resize(img, (cfg.TEST_DATA_SIZE, cfg.TEST_DATA_SIZE))
        net.blobs['data'].data[...] = np.array(data_img).transpose((2, 0, 1))
        net.forward()
        pred = net.blobs['prob'].data.flatten()
        sorted_pred_idx = np.argsort(pred)[::-1]
        real_cls_id = real_big_class[sorted_pred_idx[0]]
        cls_name = cls_names[real_cls_id]
        print("Classification result for %s: %s, accuracy = %f\n" % (i, cls_name, np.max(pred)))


        # Step 2: extract deep features
        bboxes = []
        img_scales = np.ones(1)
        img_shape = img.shape
        bboxes.append(np.zeros(4))
        bboxes[0][0] = 0
        bboxes[0][1] = 0
        bboxes[0][2] = img_shape[1] * img_scales
        bboxes[0][3] = img_shape[0] * img_scales
        fc_vec, pool_vec = extract_ftr(net_roi, img, bboxes)
        #######################################################################################


        # Step 3: compress deep features using PCA
        n, c, h, w = pool_vec.shape
        fc_vec = fc_vec.reshape((fc_vec.shape[0] * fc_vec.shape[1]))
        pool_vec = pool_vec.reshape((n * c * h * w))
        fc_list = fc_vec.tolist()
        pool_list = pool_vec.tolist()
        [fc_pca, fc_k, pool_pca, pool_k] = pca_func(fc_list, pool_list, opt.imagenet_path, cls_name, nargout=4)
        #######################################################################################


        # Step 4: search similar images in ImageNet using compressed deep features
        data_fc = fc_pca._data
        data_pool = pool_pca._data
        arr_fc = (c_float * len(data_fc))(*data_fc)
        arr_pool = (c_float * len(data_pool))(*data_pool)
        search_func(byref(arr_fc), byref(arr_pool), opt.imagenet_path, opt.in_path, opt.out_path, i, cls_name,
                    cfg.TEST_MIN_SIZE, cfg.POOL_FTR_LEN, c_float(cfg.FTR_WEIGHT), c_float(cfg.HIST_WEIGHT))

        # done
