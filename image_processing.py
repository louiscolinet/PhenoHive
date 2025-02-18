"""
Script python qui process les images prisent par la caméra pour évaluer la croissance des plantes
"""
from plantcv import plantcv as pcv
import numpy as np
import datetime
import cv2


def get_height_pix(image_path: str, pot_limit: int, channel: str = 'k', kernel_size: int = 3,
                   fill_size: int = 1) -> int:
    """
    Get the height of the plant in pixel
    :param image_path: path to the image
    :param pot_limit: height of the pot in pixels
    :param channel: CMYK channel for conversion from RGB to CMYK colorspace
    (c = cyan, m = magenta, y = yellow, k=black)
    :param kernel_size: kernel size for the median blur
    :param fill_size: PCV will identify objects in the image and fills those that are less than size
    :return: the height of the plant in pixels
    """
    date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    path = "data/images/edges_img/edge%s.jpg" % date
    pcv.params.debug = None

    img, path, _ = pcv.readimage(image_path)

    height, width = img.shape[0], img.shape[1]

    k = pcv.rgb2gray_cmyk(rgb_img=img, channel=channel)
    k_mblur = pcv.median_blur(k, kernel_size)

    edges = pcv.canny_edge_detect(k_mblur, sigma=2)
    edges_crop = pcv.crop(edges, 5, 5, height - pot_limit - 10, width - 10)
    new_height = edges_crop.shape[0]
    edges_filled = pcv.fill(edges_crop, fill_size)
    pcv.print_image(edges_filled, path)
    non_zero = np.nonzero(edges_filled)
    # height = position of the last non-zero pixel
    plant_height_pix = new_height - min(non_zero[0])

    return plant_height_pix


def get_segment_list(image_path: str, channel: str = 'k', kernel_size: int = 20) -> list[int]:
    """
    Get the list of segments lengths from the plant skeleton
    :param image_path: path to the image
    :param channel: CMYK channel for conversion from RGB to CMYK colorspace
    (c = cyan, m = magenta, y = yellow, k=black)
    :param kernel_size: kernel size for the closing operation
    :raises: KeyError if no segments are found in the image
    :return: list of segments lengths
    """
    pcv.params.debug = None

    # Read image
    img, _, _ = pcv.readimage(image_path)

    # Get image dimension
    height, width = img.shape[0], img.shape[1]

    # Extract channel (grey image)
    k = pcv.rgb2gray_cmyk(rgb_img=img, channel=channel)

    # Perform canny=edge detection
    edges = pcv.canny_edge_detect(k, sigma=2)

    # Crop image edges
    edges_crop = pcv.crop(edges, 5, 5, height - 10, width - 10)

    # Close gaps in plant contour
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(edges_crop, cv2.MORPH_CLOSE, kernel)

    # Find contours
    thresh = cv2.threshold(closing, 128, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]
    big_contour = max(contours, key=cv2.contourArea)

    # Fill contour to get maize shape
    result = np.zeros_like(closing)
    cv2.drawContours(result, [big_contour], 0, (255, 255, 255), cv2.FILLED)

    # Draw plant skeleton and segment
    pcv.params.line_thickness = 3
    skeleton = pcv.morphology.skeletonize(mask=result)
    segmented_img, obj = pcv.morphology.segment_skeleton(skel_img=skeleton)

    if pcv.params.debug is not None:
        # The labelled image is only useful for debugging purposes
        _ = pcv.morphology.segment_path_length(segmented_img=segmented_img,
                                               objects=obj, label="default")
    # Get segment lengths
    # Will raise a KeyError if no segments are found
    path_lengths = pcv.outputs.observations['default']['segment_path_length']['value']
    return path_lengths


def get_total_length(image_path: str, channel: str = 'k', kernel_size: int = 20) -> int:
    """
    Get the total length of the plant skeleton
    :param image_path: path to the image
    :param channel: CMYK channel for conversion from RGB to CMYK colorspace
    (c = cyan, m = magenta, y = yellow, k=black)
    :param kernel_size: kernel size for the closing operation
    :raises: KeyError if no segments are found in the image
    :return: the total length of the plant skeleton
    """
    # May raise a KeyError if no segments are found
    segment_list = get_segment_list(image_path, channel, kernel_size)

    # Get the sum of segment lengths
    return sum(segment_list)
