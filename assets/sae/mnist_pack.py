# save as make_mnist_pack.py
import json, numpy as np
from tensorflow.keras.datasets import mnist

# how many per digit
K = 20
# normalize to [0,1] floats
def prep(x):
    x = x.astype(np.float32)/255.0
    return x.reshape(-1, 28*28).tolist()

(_, _), (x, y) = mnist.load_data()
# (Or use train split; either is fine for a demo)
pack = {}
for d in range(10):
    idx = np.where(y==d)[0][:K]
    pack[str(d)] = prep(x[idx])

with open('mnist_digits.json', 'w') as f:
    json.dump(pack, f)
print('wrote mnist_digits.json')