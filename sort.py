import numpy as np
from kalman import KalmanBoxTracker
from utils2 import *

class Sort(object):

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        '''
        Updates the state of the trackers on a new frame of detections.
        dets: argument on the form [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...], where each index is a detection.
        '''

        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):

            # Obtain current state estimate
            pos = self.trackers[t].predict()[0]

            # Updates the tracker with new predictions
            trk[:] = [*pos, 0]

            # Marks trackers with NaN values for deletion
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Exclude lines with NaN values / Invalid lines
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        # Remove trackers marked for deletion
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = detection_to_trackers(
            dets, trks, self.iou_threshold)

        # Updates the Kalman filters associated with the trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Creates new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        counter = len(self.trackers)

        # Iterates over the trackers to generate outputs and remove the old ones
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            
            # If the tracker has been recently updated and its hit streak is greater than the minimum number of hits, it is considered valid
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):

                # Appends the result of the tracker to the output
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))

            counter -= 1

            # Remove old trackers if they have not been updated for too long
            if trk.time_since_update > self.max_age:
                self.trackers.pop(counter)

        if not not ret:
            return np.concatenate(ret)
        return np.empty((0, 5))
