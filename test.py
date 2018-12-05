import numpy as np
import pickle
from skimage import io
from m_function import normSize, img2grey, haralick, hu_moments


class Elemento:
    def __init__(self):
        self.label = None
        self.image = None
        self.feature = []
        self.distance = 0


with open('means.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    b_mean, o_mean, l_mean = pickle.load(f)

# IMPORT Y ANALISIS DE IMAGEN A TESTEAR
test = Elemento()
test_dir = input("Enter test file name: ")
aux = io.imread('./test/' + test_dir)
aux = normSize(aux)
test.image = img2grey(aux, mode='cv')

test_fht = haralick(test.image)
test_fhm = hu_moments(test.image)
aux = np.hstack([test_fht, test_fhm])
test.feature = aux.reshape(1, -1)

test.label = 'banana'

# RESULTADO CALCULANDO DISTANCIAS A LOS MEANS FINALES
sum_b = 0
sum_o = 0
sum_l = 0

for i in range(0, len(test.feature[0])-1):
    sum_b += np.power(np.abs(test.feature[0][i] - b_mean[i]), 2)
    sum_o += np.power(np.abs(test.feature[0][i] - o_mean[i]), 2)
    sum_l += np.power(np.abs(test.feature[0][i] - l_mean[i]), 2)

dist_b = np.sqrt(sum_b)
dist_o = np.sqrt(sum_o)
dist_l = np.sqrt(sum_l)
print(dist_b, dist_o, dist_l)

if ((dist_b <= dist_o) and (dist_b <= dist_l)):
    test.label = 'banana'
if ((dist_o <= dist_b) and (dist_o <= dist_l)):
    test_label = 'orange'
if ((dist_l <= dist_b) and (dist_l <= dist_o)):
    test.label = 'lemon'

print(test.label)
