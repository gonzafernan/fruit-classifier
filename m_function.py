
from skimage import color, filters

import cv2
import mahotas


# Conversión de imagen a escala de grises
def img2grey(image, mode='sk'):
    if (mode == 'sk'):
        gray = color.rgb2gray(image)
    elif (mode == 'cv'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


# Normalización del tamaño de la imagen
def normSize(image, size=(tuple((100, 100)))):
    image = cv2.resize(image, size)
    return image


# Filtrado de imagen con filtro gaussiano
def imgClean(image, sigma=1):
    clean = filters.gaussian(image, sigma)
    return clean


# Detección de bordes con filtro sobel
def imgEdge(image, sigma=1):
    aux = imgClean(image, sigma)
    edge = filters.sobel(aux)
    return edge


# Extracción de características Hu Moments
def hu_moments(image):
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# Extracción de características Haralick Textures
def haralick(image):
    feature = mahotas.features.haralick(image).mean(axis=0)
    return feature


class Elemento():
    def __init__(self):
        self.label = None
        self.image = None
        self.feature = []
        self.distance = 0
