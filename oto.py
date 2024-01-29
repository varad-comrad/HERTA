import math
import cv2
from ultralytics import YOLO
from sort import Sort
import numpy as np

video = cv2.VideoCapture('gado.mp4')
classes = ['Dog', 'Koala', 'Zebra', 'pig', 'antelope', 'badger', 'bat', 'bear', 'bison', 'cat', 'chimpanzee',
           'cow', 'coyote', 'deer', 'donkey', 'duck', 'eagle', 'elephant', 'flamingo', 'fox', 'goat', 'goldfish',
           'goose', 'gorilla', 'hamster', 'horse', 'human', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo',
           'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'pigeon',
           'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'peguin', 'pelecaniformes', 'porcupine',
           'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake',
           'starfish', 'swain', 'tiger', 'turkey', 'turtle', 'undetected', 'whale', 'whale-shark', 'wolf', 'woodpecker']

class OTO:
    '''
    The tracker class which the user will interact with. Configures the settings for the tracker and tracks the objects in the video.
    '''
    def __init__(self, video,
                       classes, 
                       max_age=20,
                       min_hits=3, 
                       detection_model='yolo', 
                       iou_threshold=0.3, 
                       language='en'):
        
        import pygame
        if detection_model == 'yolo':
            self.detection_model = YOLO('yolov8n.pt')
        else:
            raise NotImplementedError('Model not yet implemented.')
        self.classes = classes
        self.cap = video
        self.iou_threshold = iou_threshold
        self.lang = language
        self.max_age = max_age
        self.min_hits = min_hits
        self._legend = False
        self.limits = [400,400,700,700]
        self.limitT = [400,400,700,700]
        self.totalCount = []

    def set(self, **kwargs):
        self.__dict__.update(kwargs)

    def legend(self, flag=True):
        self._legend = flag

    def init(self):
        self.tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)

    def run(self):
        self.init()
        while True:
            success, img = self.cap.read()
            if not success:
                break
            results = self.detection_model(img, stream=True)
            detections = np.empty((0,5))
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    detections = np.append(detections, [[x1, y1, x2, y2, 0]], axis=0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    w, h = x2-x1, y2-y1
                    conf = math.ceil((box.conf[0]*100))/100
            resultsTracker = self.tracker.update(detections)
            for resultado in resultsTracker:
                x1, y1, x2, y2, id = resultado
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(resultado)
            try:
                cv2.imshow("Image", img)
            except:
                break
            cv2.waitKey(1)
