import sys

import pickle
from skimage import io
from m_function import Elemento, printProgressBar, ft_extract

dir = sys.argv[1]

# IMPORT DE LA BASE DE DATOS
banana = io.ImageCollection(
    './../' + dir + '/banana/*.png:./../' + dir + '/banana/*.jpg')
orange = io.ImageCollection(
    './../' + dir + '/orange/*.png:./../' + dir + '/orange/*.jpg')
lemon = io.ImageCollection(
    './../' + dir + '/lemon/*.png:./../' + dir + '/lemon/*.jpg')

# ANALISIS DE LA BASE DE DATOS
data = []
i = 0
# Análisis de bananas en base de datos
iter = 0
printProgressBar(iter, len(banana), prefix='Loading banana data:',
                 suffix='Complete', length=50)
for fruit in banana:
    data.append(Elemento())
    data[i].label = 'banana'
    data[i].image, data[i].feature = ft_extract(fruit)
    # print(data[i].feature.shape)
    i += 1
    iter += 1
    printProgressBar(iter, len(banana), prefix='Loading banana data:',
                     suffix='Complete', length=50)
print("Banana data is ready")

# Análisis de naranjas en base de datos
iter = 0
printProgressBar(iter, len(orange), prefix='Loading orange data:',
                 suffix='Complete', length=50)
for fruit in orange:
    data.append(Elemento())
    data[i].label = 'orange'
    data[i].image, data[i].feature = ft_extract(fruit)
    i += 1
    iter += 1
    printProgressBar(iter, len(orange), prefix='Loading orange data:',
                     suffix='Complete', length=50)
print("Orange data is ready")

# Análisis de limones en la base de datos
iter = 0
printProgressBar(iter, len(lemon), prefix='Loading lemon data:',
                 suffix='Complete', length=50)
for fruit in lemon:
    data.append(Elemento())
    data[i].label = 'lemon'
    data[i].image, data[i].feature = ft_extract(fruit)
    i += 1
    iter += 1
    printProgressBar(iter, len(lemon), prefix='Loading lemon data:',
                     suffix='Complete', length=50)
print("Lemon data is ready")

f = open(dir + '.pkl', 'wb')
pickle.dump(data, f)
f.close()
print("Data analysis completed")
