import numpy as np
import torch
import torchvision.transforms as T
from skimage import io, transform

from InceptionUNET.config import *
from InceptionUNET.model import Unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = Unet(MODEL_IN, MODEL_OUT)
unet = unet.to(device)
unet.load_state_dict(torch.load(WEIGHTS_FILE))
unet.eval()

img_transform = T.Compose([
    T.Normalize((0.807364995678337, 0.8255694485876925, 0.8081173673702616, 0.5, 0.5),
                (0.20619193505040878, 0.21304207788633844, 0.2254893851228971, 0.2886729, 0.2886741))
])


# size of image the 384 x 256
def coordinate_map(shape):
    h, w = shape
    arr0 = np.zeros((h, w))
    arr0 += range(arr0.shape[1])

    arr1 = np.zeros((w, h))
    arr1 += range(arr1.shape[1])
    arr1 = arr1.T

    arr = np.dstack((arr0, arr1))

    return arr


def toTensor(image):
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).float()
    image[:3] = image[:3] / 255
    image[3] = image[3] / 256
    image[4] = image[4] / 384
    image = img_transform(image)

    return image.unsqueeze(0)


def predict(filename):
    image = io.imread(filename)

    shape = (384, 256)
    image = transform.resize(image, output_shape=shape, anti_aliasing=True)

    image_with_coord = np.zeros((shape[0], shape[1], 5), dtype='uint8')
    image_with_coord[:, :, :3] = image
    image_with_coord[:, :, 3:5] = coordinate_map(shape)

    tensor_image = toTensor(image_with_coord)
    output = unet(tensor_image)

    return output
