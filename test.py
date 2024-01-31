import cv2
from oto import OTO

video = cv2.VideoCapture('highway0.mp4')

OTO(video,['truck', 'car', 'bus']).legend().enable_counter().run()