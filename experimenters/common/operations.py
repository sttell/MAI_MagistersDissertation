import numpy as np
import cv2


def _calculate_resize_shape(data_shape, major_size=64):
    if len(data_shape) == 2:
        height, width = data_shape
    else:
        height, width, _ = data_shape

    ratio = max(width, height) / min(width, height)
    minor_size = int(round(major_size / ratio, 0))

    dst_w, dst_h = 0, 0
    if width > height:
        dst_w = major_size
        dst_h = minor_size
    else:
        dst_w = minor_size
        dst_h = major_size
    return dst_w, dst_h


def apply_bilateral_filter(image, params):
    return cv2.bilateralFilter(image, params["d"], params["sigma_color"], params["sigma_space"])


def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_resize(image, params):
    resized = cv2.resize(
        image,
        _calculate_resize_shape(image.shape, params["major_size"]),
        interpolation=params["interpolation"]
    )
    if params['convert_square']:
        w_padding = params["major_size"] - resized.shape[0]
        h_padding = params["major_size"] - resized.shape[1]
        resized = cv2.copyMakeBorder(resized, top=w_padding // 2, bottom=w_padding // 2, right=h_padding // 2,
                                     left=h_padding // 2, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return resized


def apply_sobel(image, params):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=params["ksize"])
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=params["ksize"])
    sobel = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    return (sobel + np.min(sobel)) / np.max(sobel) * 254.0


def apply_flatten(image):
    return image.flatten()
