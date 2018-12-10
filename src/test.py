import numpy as np
import pickle
from skimage import io
from m_function import Elemento, ft_extract


with open('means.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    b_mean, o_mean, l_mean = pickle.load(f)

# IMPORT Y ANALISIS DE IMAGEN A TESTEAR
test = Elemento()
test_dir = input("Enter test file name: ")
image = io.imread('./test/' + test_dir)

test.image, test.feature = ft_extract(image)

test.label = 'banana'

# RESULTADO CALCULANDO DISTANCIAS A LOS MEANS FINALES
sum_b = 0
sum_o = 0
sum_l = 0

for i in range(0, len(test.feature)-1):
    sum_b += np.power(np.abs(test.feature[i] - b_mean[i]), 2)
    sum_o += np.power(np.abs(test.feature[i] - o_mean[i]), 2)
    sum_l += np.power(np.abs(test.feature[i] - l_mean[i]), 2)

dist_b = np.sqrt(sum_b)
dist_o = np.sqrt(sum_o)
dist_l = np.sqrt(sum_l)

if ((dist_b <= dist_o) and (dist_b <= dist_l)):
    test.label = 'banana'
if ((dist_o <= dist_b) and (dist_o <= dist_l)):
    test_label = 'orange'
if ((dist_l <= dist_b) and (dist_l <= dist_o)):
    test.label = 'lemon'

print(test.label)
