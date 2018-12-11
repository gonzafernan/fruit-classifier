import pickle
from skimage import io
from m_function import Elemento, printProgressBar, ft_extract

banana_test = io.ImageCollection(
    './../test/banana/*.png:./../test/banana/*.jpg')
orange_test = io.ImageCollection(
    './../test/orange/*.png:./../test/orange/*.jpg')
lemon_test = io.ImageCollection(
    './../test/lemon/*.png:./../test/lemon/*.jpg')

test = []
i = 0

# Análisis de bananas en base de datos
iter = 0
printProgressBar(iter, len(banana_test), prefix='Loading banana test data:',
                 suffix='Complete', length=50)
for fruit in banana_test:
    test.append(Elemento())
    test[i].label = 'banana'
    test[i].image, test[i].feature = ft_extract(fruit)
    # print(data[i].feature.shape)
    i += 1
    iter += 1
    printProgressBar(iter, len(banana_test),
                     prefix='Loading banana test data:',
                     suffix='Complete', length=50)
print("Banana test data is ready")

# Análisis de naranjas en base de datos
iter = 0
printProgressBar(iter, len(orange_test), prefix='Loading orange test data:',
                 suffix='Complete', length=50)
for fruit in orange_test:
    test.append(Elemento())
    test[i].label = 'orange'
    test[i].image, test[i].feature = ft_extract(fruit)
    i += 1
    iter += 1
    printProgressBar(iter, len(orange_test),
                     prefix='Loading orange test data:',
                     suffix='Complete', length=50)
print("Orange test data is ready")

# Análisis de limones en la base de datos
iter = 0
printProgressBar(iter, len(lemon_test), prefix='Loading lemon test data:',
                 suffix='Complete', length=50)
for fruit in lemon_test:
    test.append(Elemento())
    test[i].label = 'lemon'
    test[i].image, test[i].feature = ft_extract(fruit)
    i += 1
    iter += 1
    printProgressBar(iter, len(lemon_test),
                     prefix='Loading lemon test data:',
                     suffix='Complete', length=50)
print("Lemon test data is ready")

f = open('test.pkl', 'wb')
pickle.dump(test, f)
f.close()

print("Test data analysis completed")
