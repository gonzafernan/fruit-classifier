import numpy as np
import random
import pickle
from m_function import printProgressBar


f = open('data.pkl', 'rb')
data = pickle.load(f)
f.close()

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

iter = 0
MAX_ITER = 50

printProgressBar(iter, MAX_ITER, prefix='Progress:',
                 suffix='Complete', length=50)

while (iter < MAX_ITER):

    banana_data = []
    orange_data = []
    lemon_data = []

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
    """
    i = 0
    for i in range(0, len(data[0].feature[0])-1):
        sum_b = 0
        for b in banana_data:
            sum_b += b[i]
        sum_o = 0
        for o in orange_data:
            sum_o += o[i]
        sum_l = 0
        for l in lemon_data:
            sum_l += l[i]

        b_mean[i] = sum_b / len(banana_data)
        o_mean[i] = sum_o / len(orange_data)
        l_mean[i] = sum_l / len(lemon_data)
    """
    sum_b = np.zeros(b_mean.shape, b_mean.dtype)
    sum_o = np.zeros(o_mean.shape, o_mean.dtype)
    sum_l = np.zeros(l_mean.shape, l_mean.dtype)

    for b in banana_data:
        sum_b += b
    for o in orange_data:
        sum_o += o
    for l in lemon_data:
        sum_l += l

    b_mean = sum_b / len(banana_data)
    o_mean = sum_o / len(orange_data)
    l_mean = sum_l / len(lemon_data)

    # CONDICION DE SALIDA (Por ahora por cantidad de iteraciones)
    iter += 1
    printProgressBar(iter, MAX_ITER, prefix='Progress:',
                     suffix='Complete', length=50)
    # CONVERGENCIA
    # print(len(banana_data), len(orange_data), len(lemon_data))

with open('means.pkl', 'wb') as f:
    pickle.dump([b_mean, o_mean, l_mean], f)
