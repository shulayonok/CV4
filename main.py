import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import lines
import circles

# Считываем изображение
image = np.array(Image.open("brain21.jpg"))

image1 = lines.apply(image)
plt.imshow(image)
plt.show()

circles.apply(image, image1)
plt.imshow(image)
plt.show()
