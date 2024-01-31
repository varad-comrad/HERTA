import math
from typing import Self
import cv2
from ultralytics import YOLO
from sort import Sort
import numpy as np
import time


class OTO:
    '''
    The tracker class which the user will interact with. Configures the settings for the tracker and tracks the objects in the video.
    '''
    

    cls = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
        9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
        33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
        58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
        66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }

    rev_cls = {value: key for key, value in cls.items()}

    def __init__(self, video,
                       classes=None, 
                       max_age=20,
                       min_hits=3, 
                       iou_threshold=0.3, 
                       language='en',
                       legend=False,
                       counter=False,
                       bbox_color=(255, 0, 255),
                       thickness=3):
        
        import pygame
        self.detection_model = YOLO('yolov8n.pt') 
        if not isinstance(classes, list | tuple | None):
            raise TypeError('Classes must be a list or a tuple.')
        self.classes = classes
        self.cap = video
        self.iou_threshold = iou_threshold
        self.lang = language
        self.max_age = max_age
        self.min_hits = min_hits
        self._legend = legend
        self._counter = counter
        self.thickness = thickness
        self.bbox_color = bbox_color

    def set(self, **kwargs):
        self.__dict__.update(kwargs)

    def legend(self, flag=True) -> Self:
        self._legend = flag
        return self
    
    def enable_counter(self, flag=True) -> Self:
        self._counter = flag
        return self

    def _fps_counter(self):
        '''
            Returns the approximate frames per second of the video
        '''
        return 1 / (time.perf_counter() - self._start_loop_timer)
    
    def _write_fps(self, img, color=(0,0,0), font_scale=0.5, thickness=1):
        '''
            Writes the frames per second on the video
        '''
        cv2.putText(img, f"FPS: {self._fps_counter()}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    def _write_label(self, img, text, coord, conf, color, font_scale=0.5, thickness=1):
        '''
            Writes the label and confidence of each detected object
        '''
        x, y = coord
        cv2.rectangle(img, (x,y-20), (x+100,y), color, -1)
        cv2.putText(img, text, (x+10,y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        cv2.putText(img, str(conf), (x+55,y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA)

    def _write_counter(self, img, color=(0,0,0), font_scale=0.5, thickness=1):
        '''
            Writes the counter of total objects detected
        '''
        pass

    def init(self):
        self.tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)
        self.frame_count = 0

    def run(self):
        self.init()
        while True:
            self._start_loop_timer = time.perf_counter()
            self.frame_count += 1
            success, img = self.cap.read()
            if not success:
                break
            results = self.detection_model(img, stream=True)
            detections = np.empty((0,5))
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if self.classes and self.cls[int(box.cls[0].item())] not in self.classes:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), self.bbox_color, self.thickness)
                    if self._legend:
                        self._write_label(img, self.cls[int(box.cls[0].item())], (x1,y1), round(box.conf[0].item(), 2), self.bbox_color)
                    w, h = x2-x1, y2-y1
                    conf = math.ceil((box.conf[0]*100))/100
                    detections = np.append(detections, [[x1, y1, x2, y2, 0]], axis=0)
            resultsTracker = self.tracker.update(detections)
            for resultado in resultsTracker:
                x1, y1, x2, y2, id = resultado
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            try:
                self._write_fps(img)
                if self._counter:
                    self._write_counter(img)
                cv2.imshow(f"Image", img)
                
            except:
                break
            cv2.waitKey(1)
