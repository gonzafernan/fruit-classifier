import numpy as np
import random
import pickle
from skimage import io
from m_function import normSize, img2grey, haralick, hu_moments


class Elemento:
    def __init__(self):
        self.label = None
        self.image = None
        self.feature = []
        self.distance = 0


# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='',
                     decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = (
        "{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


# IMPORT DE LA BASE DE DATOS
banana = io.ImageCollection('./data/banana/*.png:./data/banana/*.jpg')
orange = io.ImageCollection('./data/orange/*.png:./data/orange/*.jpg')
lemon = io.ImageCollection('./data/lemon/*.png:./data/lemon/*.jpg')

data = []
i = 0

# Análisis de bananas en base de datos
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

# Análisis de naranjas en base de datos
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

# Análisis de limones en la base de datos
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

# Means iniciales (por ahora no random)
b_flag = True
o_flag = True
l_flag = True
i = 0

# while (b_flag or o_flag or l_flag):
#     if (data[i].label == 'banana' and b_flag):
#         b_mean = data[i].feature[0]
#         b_flag = False
#     if (data[i].label == 'orange' and o_flag):
#         o_mean = data[i].feature[0]
#         o_flag = False
#     if (data[i].label == 'lemon' and l_flag):
#         l_mean = data[i].feature[0]
#         l_flag = False
#     i += 1
b_mean = random.choice(data).feature[0]
o_mean = random.choice(data).feature[0]
l_mean = random.choice(data).feature[0]

banana_data = []
orange_data = []
lemon_data = []

iter = 0
MAX_ITER = 1000

printProgressBar(iter, MAX_ITER, prefix='Progress:',
                 suffix='Complete', length=50)

while (iter < MAX_ITER):

    # ASIGNACION

    for element in data:
        sum_b = 0
        sum_o = 0
        sum_l = 0

        i = 0
        for feature in (element.feature[0]):
            sum_b += np.power(np.abs(b_mean[i] - feature), 2)
            sum_o += np.power(np.abs(o_mean[i] - feature), 2)
            sum_l += np.power(np.abs(l_mean[i] - feature), 2)
            i += 1

        dist_b = np.sqrt(sum_b)
        dist_o = np.sqrt(sum_o)
        dist_l = np.sqrt(sum_l)

        if ((dist_b <= dist_o) and (dist_b <= dist_l)):
            element.label = 'banana'
            banana_data.append(element.feature[0])
        elif ((dist_o <= dist_b) and (dist_o <= dist_l)):
            element.label = 'orange'
            orange_data.append(element.feature[0])
        elif ((dist_l <= dist_b) and (dist_l <= dist_o)):
            element.label = 'lemon'
            lemon_data.append(element.feature[0])

    # ACTUALIZACION

    i = 0

    sum_b = 0
    sum_o = 0
    sum_l = 0

    for i in range(0, len(data[0].feature[0])-1):

        for b in banana_data:
            sum_b += b[i]
        for o in orange_data:
            sum_o += o[i]
        for l in lemon_data:
            sum_l += l[i]

        b_mean[i] = sum_b / len(banana_data)
        o_mean[i] = sum_o / len(orange_data)
        l_mean[i] = sum_l / len(lemon_data)

    # CONDICION DE SALIDA (Por ahora por cantidad de iteraciones)
    iter += 1
    printProgressBar(iter, MAX_ITER, prefix='Progress:',
                     suffix='Complete', length=50)

print("\n")
with open('means.pkl', 'wb') as f:
    pickle.dump([b_mean, o_mean, l_mean], f)
