import numpy as np
import pickle
from skimage import io
from m_function import normSize, img2grey, haralick, hu_moments
from m_function import Elemento, printProgressBar


# IMPORT DE LA BASE DE DATOS
banana = io.ImageCollection('./data/banana/*.png:./data/banana/*.jpg')
orange = io.ImageCollection('./data/orange/*.png:./data/orange/*.jpg')
lemon = io.ImageCollection('./data/lemon/*.png:./data/lemon/*.jpg')

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
    aux = fruit
    aux = normSize(aux)
    data[i].image = img2grey(aux, mode='cv')

    ft_haralick = haralick(data[i].image)
    ft_hu_moments = hu_moments(data[i].image)
    global_ft = np.hstack([ft_haralick, ft_hu_moments])

    data[i].feature = global_ft.reshape(1, -1)
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
    aux = fruit
    aux = normSize(aux)

    data[i].image = img2grey(aux, mode='cv')

    ft_haralick = haralick(data[i].image)
    ft_hu_moments = hu_moments(data[i].image)
    global_ft = np.hstack([ft_haralick, ft_hu_moments])

    data[i].feature = global_ft.reshape(1, -1)
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
    aux = fruit
    aux = normSize(aux)
    data[i].image = img2grey(aux, mode='cv')

    ft_haralick = haralick(data[i].image)
    ft_hu_moments = hu_moments(data[i].image)
    global_ft = np.hstack([ft_haralick, ft_hu_moments])

    data[i].feature = global_ft.reshape(1, -1)
    i += 1
    iter += 1
    printProgressBar(iter, len(lemon), prefix='Loading lemon data:',
                     suffix='Complete', length=50)
print("Lemon data is ready")

f = open('data.pkl', 'wb')
pickle.dump(data, f)
f.close()
print("Data analysis completed")
