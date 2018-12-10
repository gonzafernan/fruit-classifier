import numpy as np
import pickle
from skimage import io
from m_function import ft_extract, Elemento

# IMPORT Y ANALISIS DE IMAGEN A TESTEAR
test = Elemento()
test_dir = input("Enter test file name: ")
image = io.imread('./../test/' + test_dir)

test.image, test.feature = ft_extract(image)

test.label = 'banana'

f = open('data.pkl', 'rb')
data = pickle.load(f)
f.close()

i = 0
sum = 0
for ft in data[0].feature:
        sum = sum + np.power(np.abs(test.feature[i] - ft), 2)
        i += 1
d = np.sqrt(sum)

for element in data:
    sum = 0
    i = 0
    for ft in (element.feature):
        sum = sum + np.power(np.abs((test.feature[i]) - ft), 2)
        i += 1

    element.distance = np.sqrt(sum)

    if (sum < d):
        d = sum
        test.label = element.label

print("K = 1")
print(test.label)

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

k = 10
print("K =", k)
for i in range(k):
    print(data[i].label)
