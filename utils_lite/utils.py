import numpy as np
import cv2

def bboxes_iou(xyxy1, xyxy2):
    """
    Calculate IoU for two bounding boxes
    :param xyxy1: array-like, contains (x1, y1, x2, y2)
    :param xyxy2: array-like, contains (x1, y1, x2, y2)
    :return: float, IoU(xyxy1, xyxy2)
    """
    x1_d, y1_d, x2_d, y2_d = xyxy1
    x1_e, y1_e, x2_e, y2_e = xyxy2

    # determine the coordinates of the intersection rectangle
    x_left = max(x1_d, x1_e)
    y_top = max(y1_d, y1_e)
    x_right = min(x2_d, x2_e)
    y_bottom = min(y2_d, y2_e)

    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    bb1_area = (max(0, x2_d - x1_d) + 1) * (max(0, y2_d - y1_d) + 1)
    bb2_area = (max(0, x2_e - x1_e) + 1) * (max(0, y2_e - y1_e) + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert 0 <= iou <= 1, f'expected value in range [0, 1], got {iou}'  # double-check ourselves

    return iou
    
def get_center(bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return (x1 + x2) // 2, (y1 + y2) // 2


def bbox_area(bbox, is_xyxy=True) -> float:
    if is_xyxy:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # (x2 - x1) * (y2 - y1)
    else:
        return bbox[2] * bbox[3]  # w * h
        
def descale_contour(contour, shape):
    new_contour = contour[::-1] * shape
    return new_contour.round().astype(np.int)

def get_roi_bbox(image_size, contours_path):
	with np.load(contours_path) as data:
		staging_cnt = descale_contour(data['larger'], image_size[:2])
		roi_cnt = descale_contour(data['smaller'], image_size[:2])
	return staging_cnt, roi_cnt


def calculate_average_movement_vector(vectors):
    """Calculating average movement vector

    Args:
        vectors (list[list]): list of vectors
    
    Returns:
        np.ndarray: average movement vector
    """
    n = len(vectors) - 1
    
    starts = np.array([[get_center(p)] for p in vectors[len(vectors)-n-1:-1]])
    ends = np.array([[get_center(p)] for p in vectors[-n:]])
    return np.mean(ends - starts, axis=0)


def calculate_angle(vector):
    """Calculates vector direction

    Args:
        vector (np.ndarray): Two dimensional vector
    
    Returns:
        int: angle in degrees
    """
    radians = np.arctan2(vector[1], vector[0])
    degrees = np.rad2deg(radians)
    sign = -1 if degrees < 0 else 1
    angle = sign * (abs(degrees) % 360)
    return angle
    
def find_zone(bbox, cur_frame, zones, still_frame):
    bbox = [max(0, v) for v in bbox]
    x1, y1, x2, y2 = bbox
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    
    intersect_dict = {}
    for zone_name, zone_contour in zones.items():
        mask = np.zeros(still_frame.shape[:2], dtype=np.uint8)
        mask = cv2.drawContours(mask, zone_contour, -1, (1), cv2.FILLED)
        
        mask_cropped = mask[y1:y2, x1:x2]
        cur_frame_cropped = cur_frame[y1:y2, x1:x2, :].copy()
        still_frame_cropped = still_frame[y1:y2, x1:x2, :].copy()

        cur_frame_masked = cv2.bitwise_and(cur_frame_cropped, cur_frame_cropped, mask=mask_cropped)
        still_frame_masked = cv2.bitwise_and(still_frame_cropped, still_frame_cropped, mask=mask_cropped)

        diff = cv2.absdiff(cur_frame_masked, still_frame_masked)

        intersect_dict[zone_name] = {
            'rel_iou': mask_cropped.sum() / bbox_area(bbox),
            'color_diff': diff.sum() / mask.sum()
        }

    zones_ = [['high_left', 'high_right'], ['mid_left', 'mid_right'], ['low_left', 'low_right']]

    for zone_level in zones_:
        left_name = zone_level[0]
        right_name = zone_level[1]

        left_values = intersect_dict[left_name]
        right_values = intersect_dict[right_name]
        
        if left_values['color_diff'] > right_values['color_diff']:
            side = left_name
        else:
            side = right_name

        if intersect_dict[side]['color_diff'] > 15:
            return side

        if 'low' in left_name:
            if left_values['rel_iou'] > right_values['rel_iou']:
                side = left_name
            else:
                side = right_name

            if intersect_dict[side]['rel_iou'] > 0:
                return side

    return None
