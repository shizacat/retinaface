#!/usr/bin/env python3

"""Predict single. Format model is onnx."""

from typing import Dict, List, Union, Optional, Tuple
from itertools import product
from math import ceil

import cv2
import onnxruntime
import numpy as np


# === Box utils
# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(
        loc: np.array,
        priors: np.array,
        variances: Union[List[float], Tuple[float, float]]
    ) -> np.array:
    """Decode locations from predictions using priors to undo the encoding
    we did for offset regression at train time.
    Args:
        loc: location predictions for loc layers,
            Shape: [num_priors, 4]
        priors: Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(
        pre: np.array,
        priors: np.array,
        variances: Union[List[float], Tuple[float, float]]
    ) -> np.array:
    """Decode landmarks from predictions using priors to undo the encoding
    we did for offset regression at train time.
    Args:
        pre: landmark predictions for loc layers,
            Shape: [num_priors, 10]
        priors: Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: Variances of priorboxes
    Return:
        decoded landmark predictions
    """
    return np.concatenate(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        axis=1,
    )


# === Prior Box
def priorbox(min_sizes, steps, clip, image_size):
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        t_min_sizes = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in t_min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    # back to torch land
    output = np.array(anchors).reshape(-1, 4)
    if clip:
        output = output.clip(max=1, min=0)
    return output


# === From other
def nms(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class Model:

    def __init__(self, model_path: str, max_size: int):
        self.model = model_path
        self.max_size = max_size

        self.variance = [0.1, 0.2]
        self.prior_box = priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=(self.max_size, self.max_size),
        )

    def predict_jsons(
            self,
            image: np.array,
            confidence_threshold: float = 0.7,
            nms_threshold: float = 0.4) -> List[Dict[str, Union[List, float]]]:
        """
        Args:
            image = <numpy array with shape (height, width, 3)>
        """
        original_height, original_width = image.shape[:2]

        scale_landmarks = np.tile([self.max_size, self.max_size], 5)
        scale_bboxes = np.tile([self.max_size, self.max_size], 2)

        # prepea image
        paded = self._pad_to_size(
            target_size=(self.max_size, self.max_size),
            image=self._normalize(
                self._longest_max_size(image, self.max_size)
            )
        )
        pads = paded["pads"]
        image = self._cnt_hwc2chw(paded["image"])[None, :]

        # Predict
        session = onnxruntime.InferenceSession(self.model)
        outs = session.run(None, {session.get_inputs()[0].name: image})
        loc, conf, land = outs
        loc, conf, land = loc[0], conf[0], land[0]  # len bs is 1 (single image)

        # Post
        conf = self._softmax(conf, axis=1)

        annotations: List[Dict[str, Union[List, float]]] = []

        boxes = decode(loc, self.prior_box, self.variance)

        boxes *= scale_bboxes
        scores = conf[:, 1]

        landmarks = decode_landm(land, self.prior_box, self.variance)
        landmarks *= scale_landmarks

        # ignore low scores
        valid_index = np.where(scores > confidence_threshold)
        boxes = boxes[valid_index]
        landmarks = landmarks[valid_index]
        scores = scores[valid_index]

        # Sort from high to low
        order = (scores * -1).argsort()  # descending
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # do NMS
        keep = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep, :].astype(np.int)

        if boxes.shape[0] == 0:
            return [{"bbox": [], "score": -1, "landmarks": []}]

        landmarks = landmarks[keep]

        scores = scores[keep].astype(np.float64)
        landmarks = landmarks.reshape([-1, 2])

        unpadded = self.unpad_from_size(pads, bboxes=boxes, keypoints=landmarks)

        resize_coeff = max(original_height, original_width) / self.max_size

        boxes = (unpadded["bboxes"] * resize_coeff).astype(int)
        landmarks = (unpadded["keypoints"].reshape(-1, 10) * resize_coeff).astype(int)

        for box_id, bbox in enumerate(boxes):
            x_min, y_min, x_max, y_max = bbox

            x_min = np.clip(x_min, 0, original_width - 1)
            x_max = np.clip(x_max, x_min + 1, original_width - 1)

            if x_min >= x_max:
                continue

            y_min = np.clip(y_min, 0, original_height - 1)
            y_max = np.clip(y_max, y_min + 1, original_height - 1)

            if y_min >= y_max:
                continue

            annotations += [
                {
                    "bbox": bbox.tolist(),
                    "score": scores[box_id],
                    "landmarks": landmarks[box_id].reshape(-1, 2).tolist(),
                }
            ]

        return annotations

    def _longest_max_size(self, img: np.array, max_size: int):
        height, width = img.shape[:2]
        scale = max_size / float(max(width, height))

        if scale != 1.0:
            new_height, new_width = tuple(
                self._py3round(dim * scale) for dim in (height, width)
            )
            img = cv2.resize(
                img,
                dsize=(new_width, new_height),
                interpolation=cv2.INTER_LINEAR
            )
        return img

    def _py3round(self, number):
        """Unified rounding in all python versions."""
        if abs(round(number) - number) == 0.5:
            return int(2.0 * round(number / 2.0))

        return int(round(number))

    def _normalize(self, img: np.array):
        # const
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        max_pixel_value=255.0

        mean = np.array(mean, dtype=np.float32)
        mean *= max_pixel_value

        std = np.array(std, dtype=np.float32)
        std *= max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)

        img = img.astype(np.float32)
        img -= mean
        img *= denominator

        return img

    def _cnt_hwc2chw(self, image: np.array):
        """Convert image shape (height, width, channel) to (C, H, W)"""
        image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
        return image

    def _pad_to_size(
            self,
            target_size: Tuple[int, int],
            image: np.array,
            bboxes: Optional[np.ndarray] = None,
            keypoints: Optional[np.ndarray] = None,
        ) -> Dict[str, Union[np.ndarray, Tuple[int, int, int, int]]]:
        """Pads the image on the sides to the target_size
        Args:
            target_size: (target_height, target_width)
            image:
            bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
            keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]
        Returns:
            {
                "image": padded_image,
                "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
                "bboxes": shifted_boxes,
                "keypoints": shifted_keypoints
            }
        """
        target_height, target_width = target_size

        image_height, image_width = image.shape[:2]

        if target_width < image_width:
            raise ValueError(
                f"Target width should bigger than image_width"
                f"We got {target_width} {image_width}")

        if target_height < image_height:
            raise ValueError(
                f"Target height should bigger than image_height"
                f"We got {target_height} {image_height}")

        if image_height == target_height:
            y_min_pad = 0
            y_max_pad = 0
        else:
            y_pad = target_height - image_height
            y_min_pad = y_pad // 2
            y_max_pad = y_pad - y_min_pad

        if image_width == target_width:
            x_min_pad = 0
            x_max_pad = 0
        else:
            x_pad = target_width - image_width
            x_min_pad = x_pad // 2
            x_max_pad = x_pad - x_min_pad

        result = {
            "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
            "image": cv2.copyMakeBorder(
                image,
                y_min_pad,
                y_max_pad,
                x_min_pad,
                x_max_pad,
                cv2.BORDER_CONSTANT
            ),
        }

        if bboxes is not None:
            bboxes[:, 0] += x_min_pad
            bboxes[:, 1] += y_min_pad
            bboxes[:, 2] += x_min_pad
            bboxes[:, 3] += y_min_pad

            result["bboxes"] = bboxes

        if keypoints is not None:
            keypoints[:, 0] += x_min_pad
            keypoints[:, 1] += y_min_pad

            result["keypoints"] = keypoints

        return result

    def unpad_from_size(
            self,
            pads: Tuple[int, int, int, int],
            image: Optional[np.array] = None,
            bboxes: Optional[np.ndarray] = None,
            keypoints: Optional[np.ndarray] = None,
        ) -> Dict[str, np.ndarray]:
        """Crops patch from the center so that sides are equal to pads.
        Args:
            image:
            pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
            bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
            keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]
        Returns: cropped image
        {
                "image": cropped_image,
                "bboxes": shifted_boxes,
                "keypoints": shifted_keypoints
            }
        """
        x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads

        result = {}

        if image is not None:
            height, width = image.shape[:2]
            result["image"] = image[y_min_pad : height - y_max_pad, x_min_pad : width - x_max_pad]

        if bboxes is not None:
            bboxes[:, 0] -= x_min_pad
            bboxes[:, 1] -= y_min_pad
            bboxes[:, 2] -= x_min_pad
            bboxes[:, 3] -= y_min_pad

            result["bboxes"] = bboxes

        if keypoints is not None:
            keypoints[:, 0] -= x_min_pad
            keypoints[:, 1] -= y_min_pad

            result["keypoints"] = keypoints

        return result

    def _softmax(self, X, axis = None):
        """
        Compute the softmax of each element along an axis of X.

        Parameters
        ----------
        X: ND-Array. Probably should be floats.
        axis (optional): axis to compute values along. Default is the
            first non-singleton axis.

        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """
        y = np.atleast_2d(X)
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
        y = y - np.expand_dims(np.max(y, axis = axis), axis)
        y = np.exp(y)
        ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
        p = y / ax_sum
        if len(X.shape) == 1: p = p.flatten()
        return p
