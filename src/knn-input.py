import numpy as np
import pickle
from skimage import io
from m_function import Elemento, ft_extract

f = open('data.pkl', 'rb')
data = pickle.load(f)
f.close()

# K AJUSTADO POR EXPERIENCIA
k = 10

# IMPORT Y ANALISIS DE IMAGEN A CLASIFICAR
fruit = Elemento()
fruit_dir = input("Enter image file direction: ")
image = io.imread(fruit_dir)

fruit.image, fruit.feature = ft_extract(image)

for element in data:
    sum = 0
    i = 0
    for ft in (element.feature):
        sum = sum + np.power(np.abs((fruit.feature[i]) - ft), 2)
        i += 1

    element.distance = np.sqrt(sum)

# Bubblesort
swap = True
while (swap):
    swap = False
    for i in range(1, len(data)-1):
        if (data[i-1].distance > data[i].distance):
            aux = data[i]
            data[i] = data[i-1]
            data[i-1] = aux
            swap = True

eval = [0, 0, 0]

for i in range(0, k):

    if (data[i].label == 'banana'):
        eval[0] += 10

    if (data[i].label == 'orange'):
        eval[1] += 10

    if (data[i].label == 'lemon'):
        eval[2] += 10

aux = eval[0]
if (aux < eval[1]):
    aux = eval[1]
if (aux < eval[2]):
    aux = eval[2]

if (aux == eval[0]):
    fruit.label = 'banana'
if (aux == eval[1]):
    fruit.label = 'orange'
if (aux == eval[2]):
    fruit.label = 'lemon'

print(fruit.label)
