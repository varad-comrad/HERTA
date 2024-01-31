import cv2
from oto import OTO

video = cv2.VideoCapture('gado.mp4')
classes = ['Dog', 'Koala', 'Zebra', 'pig', 'antelope', 'badger', 'bat', 'bear', 'bison', 'cat', 'chimpanzee',
           'cow', 'coyote', 'deer', 'donkey', 'duck', 'eagle', 'elephant', 'flamingo', 'fox', 'goat', 'goldfish',
           'goose', 'gorilla', 'hamster', 'horse', 'human', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo',
           'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'pigeon',
           'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'peguin', 'pelecaniformes', 'porcupine',
           'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake',
           'starfish', 'swain', 'tiger', 'turkey', 'turtle', 'undetected', 'whale', 'whale-shark', 'wolf', 'woodpecker']

OTO(video).legend().run()