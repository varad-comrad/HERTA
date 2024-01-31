import cv2
from oto import OTO

video = cv2.VideoCapture('highway.mp4')

OTO(video,['truck', 'car']).legend().enable_counter().run()