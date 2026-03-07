#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ابزارهای پردازش تصویر
"""

import cv2
import numpy as np
from typing import Tuple


def resize_with_padding(image: np.ndarray, target_size: Tuple[int, int] = (1920, 1080)) -> Tuple[np.ndarray, dict]:
    """
    تغییر اندازه تصویر به اندازه هدف با حفظ aspect ratio و استفاده از padding

    Args:
        image: تصویر ورودی
        target_size: اندازه هدف (width, height)

    Returns:
        (padded_image, metadata): تصویر با padding و اطلاعات تبدیل
            metadata شامل:
                - scale: نسبت scale
                - pad_top, pad_left: padding از بالا و چپ
                - original_size: اندازه اصلی (w, h)
                - resized_size: اندازه بعد از resize (w, h)
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    # محاسبه scale برای حفظ aspect ratio
    scale = min(target_w / w, target_h / h)

    # اندازه جدید با حفظ aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize تصویر
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # محاسبه padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h

    # padding در دو طرف (برای center کردن)
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # اضافه کردن padding (با رنگ مشکی)
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    metadata = {
        'scale': scale,
        'pad_top': pad_top,
        'pad_left': pad_left,
        'pad_right': pad_right,
        'pad_bottom': pad_bottom,
        'original_size': (w, h),
        'resized_size': (new_w, new_h),
        'target_size': target_size
    }

    return padded, metadata


def transform_coordinates_to_padded(coords, metadata: dict) -> list:
    """
    تبدیل مختصات از تصویر اصلی به تصویر padded

    Args:
        coords: لیست نقاط [[x1, y1], [x2, y2], ...]
        metadata: اطلاعات تبدیل از resize_with_padding

    Returns:
        لیست نقاط تبدیل شده
    """
    scale = metadata['scale']
    pad_left = metadata['pad_left']
    pad_top = metadata['pad_top']

    transformed = []
    for point in coords:
        x, y = point
        new_x = int(x * scale + pad_left)
        new_y = int(y * scale + pad_top)
        transformed.append([new_x, new_y])

    return transformed


def transform_bbox_to_padded(bbox: Tuple[int, int, int, int], metadata: dict) -> Tuple[int, int, int, int]:
    """
    تبدیل bounding box از تصویر اصلی به تصویر padded

    Args:
        bbox: (x1, y1, x2, y2)
        metadata: اطلاعات تبدیل

    Returns:
        bbox تبدیل شده
    """
    x1, y1, x2, y2 = bbox
    scale = metadata['scale']
    pad_left = metadata['pad_left']
    pad_top = metadata['pad_top']

    new_x1 = int(x1 * scale + pad_left)
    new_y1 = int(y1 * scale + pad_top)
    new_x2 = int(x2 * scale + pad_left)
    new_y2 = int(y2 * scale + pad_top)

    return (new_x1, new_y1, new_x2, new_y2)


def transform_coordinates_from_padded(coords, metadata: dict) -> list:
    """
    تبدیل مختصات از تصویر padded به تصویر اصلی

    Args:
        coords: لیست نقاط در فضای padded
        metadata: اطلاعات تبدیل

    Returns:
        لیست نقاط در فضای اصلی
    """
    scale = metadata['scale']
    pad_left = metadata['pad_left']
    pad_top = metadata['pad_top']

    transformed = []
    for point in coords:
        x, y = point
        orig_x = int((x - pad_left) / scale)
        orig_y = int((y - pad_top) / scale)
        transformed.append([orig_x, orig_y])

    return transformed


def transform_bbox_from_padded(bbox: Tuple[int, int, int, int], metadata: dict) -> Tuple[int, int, int, int]:
    """
    تبدیل bounding box از تصویر padded به تصویر اصلی

    Args:
        bbox: (x1, y1, x2, y2) در فضای padded
        metadata: اطلاعات تبدیل

    Returns:
        bbox در فضای اصلی
    """
    x1, y1, x2, y2 = bbox
    scale = metadata['scale']
    pad_left = metadata['pad_left']
    pad_top = metadata['pad_top']

    orig_x1 = int((x1 - pad_left) / scale)
    orig_y1 = int((y1 - pad_top) / scale)
    orig_x2 = int((x2 - pad_left) / scale)
    orig_y2 = int((y2 - pad_top) / scale)

    return (orig_x1, orig_y1, orig_x2, orig_y2)
