import numpy as np
from m_function import printProgressBar
import pickle

f1 = open('test.pkl', 'rb')
test = pickle.load(f1)
f1.close()

f2 = open('means.pkl', 'rb')
means = pickle.load(f2)
f2.close()

b_mean = means[0]
o_mean = means[1]
l_mean = means[2]

correct = 0

printProgressBar(correct, len(test),
                 prefix='Correct predictions:',
                 suffix='Total', length=50)

for t in test:

    sum_b = 0
    sum_o = 0
    sum_l = 0

    for i in range(0, len(t.feature)-1):
        sum_b += np.power(np.abs(t.feature[i] - b_mean[i]), 2)
        sum_o += np.power(np.abs(t.feature[i] - o_mean[i]), 2)
        sum_l += np.power(np.abs(t.feature[i] - l_mean[i]), 2)

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
        label = 'banana'
    if (aux == dist_o):
        label = 'orange'
    if (aux == dist_l):
        label = 'lemon'

    if (t.label == label):
        correct += 1

    printProgressBar(correct, len(test),
                     prefix='Corrects prediction:',
                     suffix='Total', length=50)
print("\n")
