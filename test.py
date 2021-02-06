import numpy as np
import random
import torch

skipdata = np.load("data/Homer_skipgram_dataset.npy",
                   allow_pickle=True)
skipdata = skipdata.tolist()


test = random.sample(skipdata, 15)

print(f"test: {skipdata[:4]}")
test2 = random.shuffle(test)
print(f"test2: {test2}")
print(f"test: {test}")


p = torch.tensor([0, 32, 324, 544], dtype=torch.int32)
print(p[1])
x = p[1].tolist()
print(type(x))
