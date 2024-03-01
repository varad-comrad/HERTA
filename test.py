import cv2
from herta import HERTA

video = cv2.VideoCapture('highway0.mp4')

HERTA(video,['truck', 'car', 'bus'], text_color=(0,0,0)).legend().enable_counter().run()