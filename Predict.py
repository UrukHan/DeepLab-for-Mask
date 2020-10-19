# Importing Libraries
import DeepLab
import PicLoad
import keras
import cv2
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


# Function of generating an image for a given network
def gen_img(img, mod):
  in_img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  input_test_img = tf.expand_dims(in_img, 0)
  gen_img = mod.predict(input_test_img)
  gen_img = gen_img[0]
  return gen_img

# Denormalize the image. (img, 0.0, 1.0, 0, 255) .astype ('uint8')
def denorm_img(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def convert(img, countClass):
  pic = np.zeros((img.shape[0], img.shape[1]))
  for i in range(len(img)):
    for j in range(len(img[0])):
      x = 0
      max = 0
      for k in range(countClass):
        if img[i][j][k] > x:
          x = img[i][j][k]
          max = k
      pic[i][j] = max
  return pic

# Setting image size for network operation
batch_size = 1
img_size = (512, 512)
num_class = 7

# Defining paths
path_test = r'D:\NeuroNet\DeepLab\HouseMask\test'

# Defining a class from a module
picload = PicLoad.PICLOAD()

# Определение дополнительных параметров модуля
picload.basePath = path_test 

# Загрузка данных
images_test = picload.load_img(picload.basePath, img_size)

# Network definition DL
DLN = DeepLab.DL()

# Model definition 
DLmodel = DLN.get_model(img_size, num_class)

# Load weights
DLmodel = keras.models.load_model(r'D:\NeuroNet\DeepLab\Model')

for i in range(len(images_test)):
    img = images_test[i].astype(float)
    img = gen_img(img, DLmodel)
    img = convert(img, num_class)
    cv2.imwrite(os.path.join(r"D:\NeuroNet\DeepLab\Predict" , r"%d.jpg" % (i)), img)





#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img_gray_mode = cv2.imread(r'D:\NeuroNet\DeepLab\Predict\0.jpg', cv2.IMREAD_GRAYSCALE)
# cifr = cv2.imread('/content/P9.png')
# def showImg(name):
#   plt.imshow(name) #, cmap = 'gray', interpolation = 'bicubic'
#   plt.show()

# gen_img = (denorm_img(gen_img, 0.0, 1.0, 0, 255).astype('uint8'))




