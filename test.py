import os
from PIL import Image
import numpy as np
for i in range(50):
    os.chdir('embryo_dataset')

    items = os.listdir()
    np.random.shuffle(items)
    os.chdir(items[0])
    print(items[0])
    items = os.listdir()
    img = Image.open(items[0])
    print(items[0])
    print(img.width, img.height)
    os.chdir("../..")
