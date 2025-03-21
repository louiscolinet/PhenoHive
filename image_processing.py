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

    img, _, _ = pcv.readimage(image_path)

    height, width = img.shape[0], img.shape[1]

    lab = pcv.rgb2gray_lab(img, channel='b')
    gray = pcv.gaussian_blur(lab, ksize=(kernel_size, kernel_size))

    edges = pcv.canny_edge_detect(gray, sigma=2)
    edges_crop = pcv.crop(edges, 5, 5, height - pot_limit - 10, width - 10)
    new_height = edges_crop.shape[0]
    edges_filled = pcv.fill(edges_crop, fill_size)
    pcv.print_image(edges_filled, path)
    non_zero = np.nonzero(edges_filled)
    # height = position of the last non-zero pixel
    plant_height_pix = new_height - min(non_zero[0])

    return plant_height_pix


def get_segment_list(image_path: str, channel: str = 'b', kernel_size: int = 20, sigma: float = 2) -> list[int]:
    pcv.outputs.clear()
    img, _, _ = pcv.readimage(image_path)

    # Utilisation de Lab au lieu de CMYK
    lab = pcv.rgb2gray_lab(img, channel='b')

    # Seuillage adaptatif + Filtrage morphologique optimisé
    thresh = cv2.adaptiveThreshold(lab, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Détection des contours avec filtrage
    contours = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500 and cv2.boundingRect(cnt)[3] > cv2.boundingRect(cnt)[2]]
    if not filtered_contours:
        raise KeyError("Aucun contour valide détecté")

    big_contour = max(filtered_contours, key=cv2.contourArea)

    # Remplissage du masque
    result = np.zeros_like(closing)
    cv2.drawContours(result, [big_contour], 0, 255, cv2.FILLED)

    # Filtrage médian avant squelette
    smoothed = cv2.medianBlur(result, 5)
    skeleton = pcv.morphology.skeletonize(mask=smoothed)

    # Suppression des artefacts du squelette
    skeleton_pruned = pcv.morphology.prune(skel_img=skeleton, size=5)

    # Segmentation du squelette et récupération des longueurs
    segmented_img, obj = pcv.morphology.segment_skeleton(skel_img=skeleton_pruned)
    _ = pcv.morphology.segment_path_length(segmented_img=segmented_img, objects=obj, label="plant")

    try:
        path_lengths = pcv.outputs.observations["plant"]["segment_path_length"]["value"]
    except KeyError:
        raise KeyError("Impossible de récupérer les longueurs des segments")

    return path_lengths

def get_total_length(image_path: str, channel: str = 'k', kernel_size: int = 20, sigma: float = 1) -> int:
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
    segment_list = get_segment_list(image_path, channel, kernel_size, sigma)

    # Get the sum of segment lengths
    return int(sum(segment_list))
    #return get_height_pix(image_path, 5, channel, kernel_size)
