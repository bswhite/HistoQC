import logging
import os
import numpy as np
from histoqc.BaseImage import printMaskHelper
from skimage import io, color, img_as_ubyte
from distutils.util import strtobool
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
from sklearn.cluster import KMeans
from skimage import exposure

from scipy.ndimage.morphology import binary_dilation
from skimage.measure import find_contours
from skimage.draw import polygon

import matplotlib.pyplot as plt



def getIntensityThresholdOtsu(s, params):
    logging.info(f"{s['filename']} - \tLightDarkModule.getIntensityThresholdOtsu")
    name = params.get("name", "classTask")    
    local = strtobool(params.get("local", "False"))
    radius = float(params.get("radius", 15))
    selem = disk(radius)

    img = s.getImgThumb(s["image_work_size"])
    img = color.rgb2gray(img)

    if local:
        thresh = rank.otsu(img, selem)
    else:
        thresh = threshold_otsu(img)

    map = img < thresh

    s["img_mask_" + name] = map > 0
    if strtobool(params.get("invert", "False")):
        s["img_mask_" + name] = ~s["img_mask_" + name]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_" + name + ".png", img_as_ubyte(s["img_mask_" + name]))

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & s["img_mask_" + name]

    s.addToPrintList(name,
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.getIntensityThresholdOtsu:{name} NO tissue remains "
                        f"detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.getIntensityThresholdOtsu:{name} NO tissue remains detectable! "
                             f"Downstream modules likely to be incorrect/fail")

    return

def contourToMask(contour, image):
    mask = np.zeros(image.shape)
    rr, cc = polygon(contour[:, 0], contour[:, 1], mask.shape)
    mask[rr, cc] = 1
    return mask

def getIntensityThresholdContours(s, params):
    logging.info(f"{s['filename']} - \tLightDarkModule.getIntensityThresholdContours")
    name = params.get("name", "classTask")
    # radius of element to use in dilation
    radius = float(params.get("radius", 3))
    # number of iterations to apply in dilation
    iterations = int(params.get("iterations", 3))
    # minimum fractional area of (filled) regions/contours to keep in mask (0.1 = 10%)
    min_fractional_area = float(params.get("min_fractional_area", 0.1))
    selem = disk(radius)

    # this is applied to mask from previous step!
    img = s["img_mask_use"]
    img = color.rgb2gray(img)

    # dilate the mask to smooth / close small openings 
    dilated_img = binary_dilation(img, structure=selem, iterations=iterations)

    # calculate the contours
    contours = find_contours(dilated_img, 0.5)

    # calculate masks from the contours (i.e., filled regions)
    masks = [contourToMask(contour, dilated_img) for contour in contours]

    # only keep "large" masks abouve user-specified threshold
    max_area = dilated_img.shape[0] * dilated_img.shape[1]
    threshold_area = min_fractional_area * max_area
    # np.sum(mask) = area of mask
    large_masks = [mask for mask in masks if np.sum(mask) > threshold_area]

    # take the union of the masks
    map = np.logical_or.reduce(large_masks)
    
    s["img_mask_" + name] = map > 0
    if strtobool(params.get("invert", "False")):
        s["img_mask_" + name] = ~s["img_mask_" + name]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_" + name + ".png", img_as_ubyte(s["img_mask_" + name]))

    prev_mask = s["img_mask_use"]
    # note that this mask _replaces_ the previous mask, it isn't logically and'ed to it as
    # in most modules
    s["img_mask_use"] = s["img_mask_" + name]

    s.addToPrintList(name,
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.getIntensityThresholdContours:{name} NO tissue remains "
                        f"detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.getIntensityThresholdContours:{name} NO tissue remains detectable! "
                             f"Downstream modules likely to be incorrect/fail")

    return


def getIntensityThresholdPercent(s, params):
    name = params.get("name", "classTask")
    logging.info(f"{s['filename']} - \tLightDarkModule.getIntensityThresholdPercent:\t {name}")

    lower_thresh = float(params.get("lower_threshold", -float("inf")))
    upper_thresh = float(params.get("upper_threshold", float("inf")))

    lower_var = float(params.get("lower_variance", -float("inf")))
    upper_var = float(params.get("upper_variance", float("inf")))

    img = s.getImgThumb(s["image_work_size"])
    img_var = img.std(axis=2)

    map_var = np.bitwise_and(img_var > lower_var, img_var < upper_var)

    img = color.rgb2gray(img)
    map = np.bitwise_and(img > lower_thresh, img < upper_thresh)

    map = np.bitwise_and(map, map_var)

    s["img_mask_" + name] = map > 0



    if strtobool(params.get("invert", "False")):
        s["img_mask_" + name] = ~s["img_mask_" + name]

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & s["img_mask_" + name]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_" + name + ".png", img_as_ubyte(prev_mask & ~s["img_mask_" + name]))

    s.addToPrintList(name,
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.getIntensityThresholdPercent:{name} NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.getIntensityThresholdPercent:{name} NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")

    return






def removeBrightestPixels(s, params):
    logging.info(f"{s['filename']} - \tLightDarkModule.removeBrightestPixels")

    # lower_thresh = float(params.get("lower_threshold", -float("inf")))
    # upper_thresh = float(params.get("upper_threshold", float("inf")))
    #
    # lower_var = float(params.get("lower_variance", -float("inf")))
    # upper_var = float(params.get("upper_variance", float("inf")))

    img = s.getImgThumb(s["image_work_size"])
    img = color.rgb2gray(img)

    kmeans = KMeans(n_clusters=3,  n_init=1).fit(img.reshape([-1, 1]))
    brightest_cluster = np.argmax(kmeans.cluster_centers_)
    darkest_point_in_brightest_cluster = (img.reshape([-1, 1])[kmeans.labels_ == brightest_cluster]).min()

    s["img_mask_bright"] = img > darkest_point_in_brightest_cluster



    if strtobool(params.get("invert", "False")):
        s["img_mask_bright"] = ~s["img_mask_bright"]

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & s["img_mask_bright"]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_bright.png", img_as_ubyte(prev_mask & ~s["img_mask_bright"]))

    s.addToPrintList("brightestPixels",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.removeBrightestPixels NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.removeBrightestPixels NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")

    return



def minimumPixelIntensityNeighborhoodFiltering(s,params):
    logging.info(f"{s['filename']} - \tLightDarkModule.minimumPixelNeighborhoodFiltering")
    disk_size = int(params.get("disk_size", 10000))
    threshold = int(params.get("upper_threshold", 200))

    img = s.getImgThumb(s["image_work_size"])
    img = color.rgb2gray(img)
    img = (img * 255).astype(np.uint8)
    selem = disk(disk_size)

    imgfilt = rank.minimum(img, selem)
    s["img_mask_bright"] = imgfilt > threshold


    if strtobool(params.get("invert", "True")):
        s["img_mask_bright"] = ~s["img_mask_bright"]

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & s["img_mask_bright"]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_bright.png", img_as_ubyte(prev_mask & ~s["img_mask_bright"]))

    s.addToPrintList("brightestPixels",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.minimumPixelNeighborhoodFiltering NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.minimumPixelNeighborhoodFiltering NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")

    return

def saveEqualisedImage(s,params):
    logging.info(f"{s['filename']} - \tLightDarkModule.saveEqualisedImage")

    img = s.getImgThumb(s["image_work_size"])
    img = color.rgb2gray(img)

    out = exposure.equalize_hist((img*255).astype(np.uint8))
    io.imsave(s["outdir"] + os.sep + s["filename"] + "_equalized_thumb.png", img_as_ubyte(out))

    return
