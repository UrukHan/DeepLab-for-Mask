# Importing Libraries
import ImageGeneratorTF
import DeepLab
import keras
import matplotlib.pyplot as plt
import glob


# Parameter definition
img_size = (512, 512)
num_class = 7
batch_size = 2

# Defining paths
path_train_Orig = r'D:\NeuroNet\DeepLab\HouseMask\train_img'
path_train_Mask = r'D:\NeuroNet\DeepLab\HouseMask\train_mask'
path_val_Orig = r'D:\NeuroNet\DeepLab\HouseMask\val_img'
path_val_Mask = r'D:\NeuroNet\DeepLab\HouseMask\val_mask'

# Defining augmented image generator
imGen = ImageGeneratorTF.IG()

# Parameter definition for Image Generator
imGen.img_size = img_size
imGen.batch_size = batch_size

# Network training
trainGen = imGen.augmentation_DeepLab(path_train_Orig, path_train_Mask, img_size, batch_size)
valGen = imGen.augmentation_DeepLab(path_val_Orig, path_val_Mask, img_size, batch_size)

# Network definition DL
DLN = DeepLab.DL()

# Parameter definition
DLN.count = len(glob.glob(r'D:\NeuroNet\DeepLab\HouseMask\train_img\train_img\*'))

DLN.batch_size = batch_size

# Model definition 
DLmodel = DLN.get_model(img_size, num_class)

# Load weights
DLmodel = keras.models.load_model(r'D:\NeuroNet\DeepLab\Model')

# Trainig model
DLN.training(trainGen, valGen, DLmodel)


'''
for i, j in enumerate(trainGen):
    img = i
    print(img)

img = trainGenOrig[0][0][0]
img = denorm_img(img, 0.0, 1.0, 0, 255)
showImg(img.astype(int))
'''

