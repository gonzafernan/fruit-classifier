import sys

import numpy as np
import pickle
from m_function import printProgressBar


f1 = open('data.pkl', 'rb')
data = pickle.load(f1)
f1.close()

f2 = open('test.pkl', 'rb')
test = pickle.load(f2)
f2.close()

k = sys.argv[1]
k = int(k)

correct = 0

printProgressBar(correct, len(test), prefix='Correct predictions:',
                 suffix='Total', length=50)

for t in test:

    for element in data:
        sum = 0
        i = 0
        for ft in (element.feature):
            sum = sum + np.power(np.abs((t.feature[i]) - ft), 2)
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
        label = 'banana'

    if (aux == eval[1]):
        label = 'orange'

    if (aux == eval[2]):
        label = 'lemon'

    if (t.label == label):
        correct += 1
    printProgressBar(correct, len(test),
                     prefix='Correct predictions:',
                     suffix='Total', length=50)
