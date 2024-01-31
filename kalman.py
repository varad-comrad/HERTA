import numpy as np
from utils2 import bbox_to_z, z_to_bbox
from filterpy.kalman import KalmanFilter 

class KalmanBoxTracker:
    '''
    This class represents the internal state of individual tracked objects observed as bbox.
    '''
    
    cnt = 0

    def __init__(self,bbox, cls='unknown', conf=0, dim_x=7, dim_z=4):
        '''
        For this particular problem, the X dimensions are:
            0) x-coordinate of the center: Represents the horizontal position of the center of the bounding box.
            1) y-coordinate of the center: Represents the vertical position of the center of the bounding box.
            2) Scale (area) of the bounding box: Represents the size or area of the bounding box.
            3) Aspect ratio: Represents the ratio of the width to the height of the bounding box.
            4) Velocity along the x-axis: Represents the horizontal velocity of the object.
            5) Velocity along the y-axis: Represents the vertical velocity of the object.
            6) Velocity scale: Represents the rate of change of the size or area of the bounding box.
        That can be concluded by the F matrix 
        '''
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.cls = cls 
        self.conf = conf
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.cnt
        KalmanBoxTracker.cnt += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self,bbox):
        '''
        Updates the state vector with observed bbox (Zt+dt)
        '''
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox_to_z(bbox))

    def predict(self):
        '''
        Advances the state vector and returns the predicted bounding box estimate (Xt+dt).
        '''
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0

        self.kf.predict()

        self.age += 1
        self.time_since_update += 1
        
        # If a tracker has predicted twice in a row without an update, it becomes unreliable
        if self.time_since_update>0:
            self.hit_streak = 0
        
        #updates the history of the state vector of the tracker
        self.history.append(z_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        '''
        Returns the current bounding box state (Xt).
        '''
        return z_to_bbox(self.kf.x)
    
    def get_class(self):
        '''
        Returns the class associated with the tracker.
        '''
        return self.cls
