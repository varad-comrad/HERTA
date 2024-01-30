import numpy as np


def linear_alignment(cost_matrix: np.ndarray) -> np.ndarray:
    '''
    Solves the linear assignment problem using the Hungarian algorithm (Kuhn-Munkres algorithm)
    Linear assignment problem: https://en.wikipedia.org/wiki/Assignment_problem
    '''
    from scipy.optimize import linear_sum_assignment
    return np.array(list(zip(*linear_sum_assignment(cost_matrix))))


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[:, :, 0], bb_gt[:, :, 0])
    yy1 = np.maximum(bb_test[:, :, 1], bb_gt[:, :, 1])
    xx2 = np.minimum(bb_test[:, :, 2], bb_gt[:, :, 2])
    yy2 = np.minimum(bb_test[:, :, 3], bb_gt[:, :, 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    interS = w * h
    iou = interS / ((bb_test[:, :, 2] - bb_test[:, :, 0]) * (bb_test[:, :, 3] - bb_test[:, :, 1])
              + (bb_gt[:, :, 2] - bb_gt[:, :, 0]) * (bb_gt[:, :, 3] - bb_gt[:, :, 1]) - interS)
    
    return iou

def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    '''
    Returns a state representation of a bounding box for the Kalman filter
    '''

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def z_to_bbox(z: np.ndarray, score=None) -> np.ndarray:
    '''
    Returns a bounding box representation of a state of the Kalman filter
    '''
    w = np.sqrt(z[2] * z[3])
    h = z[2] / w
    if score is None:
        return np.array([z[0] - w/2., z[1] - h/2., z[0] + w/2., z[1] + h/2.]).reshape((1, 4))
    else:
        return np.array([z[0] - w/2., z[1] - h/2., z[0] + w/2., z[1] + h/2., score]).reshape((1, 5))
    
def detection_to_trackers(detections: np.ndarray, trackers: np.ndarray, iou_threshold=0.3):
    '''
    Assigns detections to tracked objects (both represented as bounding boxes)
    Returns 3 numpy arrays of matches (2D array [index_detect_bbox, index_trk_bbox]), unmatched_detections and unmatched_trackers
    '''
    if not trackers.any(): 
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    # Computes the "distance" between detections and trackers based on IoU scores
    iou_matrix = iou_batch(detections, trackers)
    
    matches_indexes = None
    if min(iou_matrix.shape) > 0: 

        # 1 if IoU of pair > iou_threshold, 0 otherwise. If it corresponds 1 to 1, simple assignment. If not, use linear assignment 
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1: 
            matches_indexes = np.stack(np.where(a), axis=1)            
        else: 
            matches_indexes = linear_alignment(-iou_matrix)
    else:
        matches_indexes = np.empty((0, 2), dtype=int)
    
    unmatched_dets = []
    unmatched_trks = []
    
    # Finds unmatched detections and trackers
    for d, det in enumerate(detections):
        if d not in matches_indexes[:, 0]:
            unmatched_dets.append(d)
    for t, trk in enumerate(trackers):
        if t not in matches_indexes[:, 1]:
            unmatched_trks.append(t)
    matches = []

    # Finds matches with low IoU scores and appends them to unmatched_dets and unmatched_trks. The rest is appended to matches
    for m in matches_indexes:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) > 0:
        matches = np.concatenate(matches, axis=0)
    else:
        matches = np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_dets), np.array(unmatched_trks)
    