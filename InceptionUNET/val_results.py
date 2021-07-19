import numpy as np
import torch
from skimage import io

from .config import *
from .dataloaders import validation_loader
from .loss_and_iou import iou
from .model import Unet


def get_output(batch_data, unet_model):
    image = batch_data["image"].cuda()
    logits = unet_model(image)
    output = torch.softmax(0.99 * torch.tanh(logits), 1)
    return output


if __name__ == '__main__':
    unet = Unet(MODEL_IN, MODEL_OUT);
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    unet = unet.to(device)

    unet.load_state_dict(torch.load(WEIGHTS_FILE))

    print("\n\nVALIDATION IMAGES\n\n")
    result_folder_name = "./Result Images/"

    unet.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            target = data["masks"].to(device)
            out = get_output(data, unet)

            for orig_image, out_mask, file_name in zip(data["no_norm"], out.cpu().numpy(), data["filename"]):
                c, h, w = out_mask.shape
                out = np.zeros((h, w, 3), dtype='uint8')
                colors = [[252, 186, 3],  # body-text-yellow
                          [252, 3, 235],  # heading -pink
                          #   [252, 150, 3], #drop-text-yellow
                          [181, 112, 255],  # floating
                          [3, 252, 103],  # caption -green
                          [252, 3, 28],  # header & footer - red
                          [19, 10, 194],  # table-darkblue
                          [3, 235, 252],  # graphic - blue
                          [0, 0, 0]]  # background-black

                color_array = np.array(colors)
                colored_mask = color_array[np.argmax(out_mask, axis=0)]

                final_img = np.concatenate((orig_image, colored_mask), axis=1)
                file_name = file_name.replace('.npy', '.png')
                io.imsave(result_folder_name + file_name, final_img)

    validation_iou = np.zeros(7, dtype=float)
    unet.eval()
    with torch.no_grad():
        for n in range(20):
            for i, data in enumerate(validation_loader):
                target = data["masks"].to(device)
                out = get_output(data, unet)

                validation_iou = validation_iou + iou(out.argmax(dim=1), target.argmax(dim=1), 8, ignore=7,
                                                      per_image=True)

    print(validation_iou / (len(validation_loader) * 20))
