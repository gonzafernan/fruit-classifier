import numpy as np
import random
import pickle
from m_function import printProgressBar


f = open('data.pkl', 'rb')
data = pickle.load(f)
f.close()

# Means iniciales
b_mean = list(random.choice(data).feature)
o_mean = list(random.choice(data).feature)
l_mean = list(random.choice(data).feature)

b_flag = True
o_flag = True
l_flag = True

b_len = 0
o_len = 0
l_len = 0

while (b_flag or o_flag or l_flag):

    banana_data = []
    orange_data = []
    lemon_data = []

    # ASIGNACION

    for element in data:
        sum_b = 0
        sum_o = 0
        sum_l = 0

        i = 0
        for feature in (element.feature):
            sum_b += np.power(np.abs(b_mean[i] - feature), 2)
            sum_o += np.power(np.abs(o_mean[i] - feature), 2)
            sum_l += np.power(np.abs(l_mean[i] - feature), 2)
            i += 1

        dist_b = np.sqrt(sum_b)
        dist_o = np.sqrt(sum_o)
        dist_l = np.sqrt(sum_l)

        if ((dist_b <= dist_o) and (dist_b <= dist_l)):
            element.label = 'banana'
            banana_data.append(element.feature)
        elif ((dist_o <= dist_b) and (dist_o <= dist_l)):
            element.label = 'orange'
            orange_data.append(element.feature)
        elif ((dist_l <= dist_b) and (dist_l <= dist_o)):
            element.label = 'lemon'
            lemon_data.append(element.feature)

    # ACTUALIZACION
    i = 0
    for i in range(0, len(data[0].feature)-1):
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

    # CONVERGENCIA Y CONDICIÓN DE SALIDA
    # print(len(banana_data), len(orange_data), len(lemon_data))
    if (len(banana_data) == b_len):
        b_flag = False
    else:
        b_len = len(banana_data)

    if (len(orange_data) == o_len):
        o_flag = False
    else:
        o_len = len(orange_data)

    if (len(lemon_data) == l_len):
        l_flag = False
    else:
        l_len = len(lemon_data)

with open('means.pkl', 'wb') as f:
    pickle.dump([b_mean, o_mean, l_mean], f)
