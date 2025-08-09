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

    k = pcv.rgb2gray_lab(rgb_img=img, channel='b')
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


def match_luminance(img: np.ndarray, background: np.ndarray, radius: int = 30) -> np.ndarray:
    """
    Adapts the background luminance to match the local luminance
    around the brightest pixel in the input image.
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    back_yuv = cv2.cvtColor(background, cv2.COLOR_RGB2YCrCb)

    img_y, _, _ = cv2.split(img_yuv)
    back_y, _, _ = cv2.split(back_yuv)

    # Find the brightest pixel
    max_loc = np.unravel_index(np.argmax(img_y), img_y.shape)
    y, x = max_loc

    # Set a window around this pixel
    y1, y2 = max(0, y - radius), min(img_y.shape[0], y + radius + 1)
    x1, x2 = max(0, x - radius), min(img_y.shape[1], x + radius + 1)

    mean_img = img_y[y1:y2, x1:x2].mean()
    mean_back = back_y[y1:y2, x1:x2].mean()

    # Adapt the luminance of the background
    back_y_norm = back_y + (mean_img - mean_back)
    back_y_norm = np.clip(back_y_norm, 0, 255).astype(np.uint8)

    back_yuv[:, :, 0] = back_y_norm
    return cv2.cvtColor(back_yuv, cv2.COLOR_YCrCb2RGB)

def remove_shadows(img: np.ndarray, beta1=0.3, beta2=0.92, tau_s=0.27, tau_h=1) -> np.ndarray:
    """
    Detect and remove shadows using HSV chromacity-based method.
    Based on Sanin et al. (2012), adapted from Cucchiara et al. (2003).

    :param img: Current RGB image (numpy array)
    :param background: Background RGB image (numpy array)
    :return: Shadow-free image (numpy array)
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

    background = cv2.imread("data/images/background.jpg")  # ou autre chemin
    print("entrée lum")
    background = match_luminance(img, background)
    print("sortie lum")
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    back_hsv = cv2.cvtColor(background, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Split channels
    Hf, Sf, Vf = cv2.split(img_hsv)
    Hb, Sb, Vb = cv2.split(back_hsv)

    # Avoid division by zero
    Vb[Vb == 0] = 1e-6

    # Compute ratios and differences
    v_ratio = Vf / Vb
    s_diff = np.abs(Sf - Sb)
    h_diff = np.abs(Hf - Hb)
    h_diff = np.minimum(h_diff, 180 - h_diff)

    # Detect shadows
    shadow_mask = (v_ratio >= beta1) & (v_ratio <= beta2) & (s_diff <= tau_s * 255) & (h_diff <= tau_h * 255)

    shadow_mask_uint8 = shadow_mask.astype(np.uint8) * 255
    shadow_mask_blurred = cv2.GaussianBlur(shadow_mask_uint8, (3, 3), 0)
    _, shadow_mask_clean = cv2.threshold(shadow_mask_blurred, 127, 255, cv2.THRESH_BINARY)
    shadow_mask = shadow_mask_clean.astype(bool)

    nb_white_pixels = np.count_nonzero(shadow_mask)

    if nb_white_pixels > 2500:
        img_no_shadow = img.copy()
        img_no_shadow[shadow_mask] = background[shadow_mask]

        # blur
        shadow_mask_u8 = (shadow_mask.astype(np.uint8) * 255)
        dilated_mask = cv2.dilate(shadow_mask_u8, np.ones((5, 5), np.uint8), iterations=1)
        transition_mask = cv2.subtract(dilated_mask, shadow_mask_u8)
        blurred = cv2.GaussianBlur(img_no_shadow, (9, 9), 0)
        img_no_shadow[transition_mask.astype(bool)] = blurred[transition_mask.astype(bool)]

        img_no_shadow = cv2.bilateralFilter(img_no_shadow, d=30, sigmaColor=45, sigmaSpace=95)
    else:
        img_no_shadow = img.copy()

    return img_no_shadow


def get_segment_list(image_path: str, channel: str = 'k', kernel_size: int = 20, sigma: float = 2, skeleton_filename = None) -> list[int]:
    """
    Get the list of segments lengths from the plant skeleton
    :param image_path: path to the image
    :param channel: CMYK channel for conversion from RGB to CMYK colorspace
    (c = cyan, m = magenta, y = yellow, k=black)
    :param kernel_size: kernel size for the closing operation
    :raises: KeyError if no segments are found in the image
    :return: list of segments lengths
    """

    # Read image
    img, _, _ = pcv.readimage(image_path)
    cv2.imwrite("data/img.jpg", img)

    # Remove shadow if any
    print("entrée_shad")
    img_no_shadow = remove_shadows(img)
    print("sortie shad")
    # Get image dimension
    height, width = img_no_shadow.shape[0], img.shape[1]

    # Extract channel (grey image)
    k = pcv.rgb2gray_cmyk(rgb_img=img_no_shadow, channel=channel)

    # Perform canny=edge detection
    edges = pcv.canny_edge_detect(k, sigma=sigma)

    # Crop image edges
    edges_crop = pcv.crop(edges, 5, 5, height - 10, width - 10)
    #cv2.imwrite("data/edges_crop.jpg", edges_crop)
    crop = pcv.crop(img, 5, 5, height - 10, width - 10)
    cv2.imwrite("data/crop.jpg", crop)

    # Close gaps in plant contour
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(edges_crop, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite("data/closing.jpg", closing)

    # Find contours
    thresh = cv2.threshold(closing, 128, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]
    if len(contours) == 0:
      return None
    big_contour = max(contours, key=cv2.contourArea)

    # Fill contour to get maize shape
    result = np.zeros_like(closing)
    cv2.drawContours(result, [big_contour], 0, (255, 255, 255), cv2.FILLED)
    cv2.imwrite("data/result.jpg", result)

    # Draw plant skeleton and segment
    pcv.params.line_thickness = 3
    skeleton = pcv.morphology.skeletonize(mask=result)
    skeleton ,segmented_img, obj = pcv.morphology.prune(skel_img=skeleton, size=20)
  
    if skeleton_filename == None : skeleton_path = "data/images/skeleton.jpg"
    else : skeleton_path = "data/images/" + skeleton_filename
    cv2.imwrite(skeleton_path, skeleton)
  
    cv2.imwrite("data/skeleton.jpg", skeleton)
    #cv2.imwrite("data/segmented_img.jpg", segmented_img)

    _ = pcv.morphology.segment_path_length(segmented_img=segmented_img,
                                               objects=obj, label="default")
    # Get segment lengths
    # Will raise a KeyError if no segments are found
    path_lengths = pcv.outputs.observations['default']['segment_path_length']['value']

    return path_lengths


def get_total_length(image_path: str, channel: str = 'k', kernel_size: int = 20, sigma: float = 1) -> float:
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
    return float(sum(segment_list))
