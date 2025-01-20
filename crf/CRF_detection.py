"""
Adapted from the inference.py to demonstate the usage of the util functions.
"""
import os
import sys

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf

# Get im{read,write} from somewhere.
try:
    from cv2 import imread, imwrite
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave
    imwrite = imsave
    # TODO: Use scipy instead.

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

# if len(sys.argv) != 4:
#     print("Usage: python {} IMAGE ANNO OUTPUT".format(sys.argv[0]))
#     print("")
#     print("IMAGE and ANNO are inputs and OUTPUT is where the result should be written.")
#     print("If there's at least one single full-black pixel in ANNO, black is assumed to mean unknown.")
#     sys.exit(1)
# print(sys.argv)
image_path="/home/iair/Downloads/dataset/test/test_image/image"
ann_path="/home/iair/Downloads/dataset/test/test_image/result"
result_path="/home/iair/Downloads/dataset/test/test_image/result_crf"
for image_name in os.listdir(image_path):
    image_p=os.path.join(image_path,image_name)
    file_name=os.path.splitext(image_name)[0]+'.png'
    ann_p=os.path.join(ann_path,file_name)
    result_p=os.path.join(result_path,file_name)
    fn_im = image_p
    fn_anno = ann_p
    fn_output = result_p
    # print(fn_anno)
    ##################################
    ### Read images and annotation ###
    ##################################
    anno_rgb = imread(fn_anno)
    anno_rgb=anno_rgb.astype(np.int32)
    # print(anno_rgb)
    img = imread(fn_im)
    # img=cv2.resize(img,[anno_rgb.shape[0],anno_rgb.shape[1]])
    # print(img.shape)
    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR

    # anno_rgb=cv2.resize(anno_rgb,[img.shape[0],img.shape[1]]).astype(np.uint32)
    # print(anno_rgb.dtype)
    # anno_rgb=np.where(anno_rgb>128,255,0)
    # print(f"anno_rgb.shape---------->{anno_rgb.shape}")
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    # print(f"annolbl.shape------------>{anno_lbl.argmax()}")
    # print(f"anno_lbl.type------------->{type(anno_lbl)}")
    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    # print(np.unique(anno_rgb))
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    print(f"colors----------->{colors}")
    print(f"labels----------->{labels.shape}")
    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0
    # if HAS_UNK:
    #     print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
    #     print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
    #     colors = colors[1:]
    #else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    # print(colorize)
    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))
    ###########################
    ### Setup the CRF model ###
    ###########################
    use_2d = False
    # use_2d = True
    if use_2d:
        print("Using 2D specialized functions")

        # Example using the DenseCRF2D code
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        print("Using generic 2D functions")

        # Example using the DenseCRF class and the util functions
        # print(img.shape)
        # print(f"n_labels------->{n_labels}")
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This creates the color-independent features and then add them to the CRF
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)


    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
    imwrite(fn_output, MAP.reshape(img.shape))

    # Just randomly manually run inference iterations
    Q, tmp1, tmp2 = d.startInference()
    for i in range(5):
        print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)
