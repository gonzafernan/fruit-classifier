import sys

import numpy as np
from skimage import io
from m_function import Elemento, ft_extract
import pickle

f = open('means.pkl', 'rb')
means = pickle.load(f)
f.close()

b_mean = means[0]
o_mean = means[1]
l_mean = means[2]


sum_b = 0
sum_o = 0
sum_l = 0

# IMPORT Y ANALISIS DE IMAGEN A CLASIFICAR
fruit = Elemento()
image = io.imread('../' + sys.argv[1])

fruit.image, fruit.feature = ft_extract(image)

for i in range(0, len(fruit.feature)-1):
    sum_b += np.power(np.abs(fruit.feature[i] - b_mean[i]), 2)
    sum_o += np.power(np.abs(fruit.feature[i] - o_mean[i]), 2)
    sum_l += np.power(np.abs(fruit.feature[i] - l_mean[i]), 2)

dist_b = np.sqrt(sum_b)
dist_o = np.sqrt(sum_o)
dist_l = np.sqrt(sum_l)
# print(dist_b, dist_o, dist_l)

aux = dist_b
if (dist_o < aux):
    aux = dist_o
if (dist_l < aux):
    aux = dist_l

if (aux == dist_b):
    fruit.label = 'banana'
if (aux == dist_o):
    fruit.label = 'orange'
if (aux == dist_l):
    fruit.label = 'lemon'

print(fruit.label)
