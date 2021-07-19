from .model import Unet
from .config import *
import torch
from torch import nn, optim
import numpy as np
from progress.bar import ChargingBar
from .dataloaders import train_loader, validation_loader
import wandb
from statistics import mean
from .loss_and_iou import dice_loss, iou


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def validation_images(data, out):
    image_array = []
    for orig_image, out_mask, file_name in zip(data["no_norm"], out.cpu().numpy(), data["filename"]):
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
        image_array.append(wandb.Image(final_img, caption=file_name))

    return image_array


def get_output(batch_data, unet_model):
    image = batch_data["image"].cuda()
    logits = unet_model(image)
    output = torch.softmax(0.99 * torch.tanh(logits), 1)
    return output


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    unet = Unet(MODEL_IN, MODEL_OUT)
    wandb.init(project="doc-seg")
    unet.apply(init_weights)

    check_path = 'checkpoint.pth'
    check_data = torch.load(check_path)
    # unet.load_state_dict(check_data["model_state_dict"])
    unet = unet.to(device)


    optimizer = optim.Adam(unet.parameters(), lr=LR, amsgrad=True)
    # optimizer.load_state_dict(check_data["optimizer_state_dict"])

    wandb.watch(unet, log='all')

    print("\n\nTRAINING\n\n")

    for epoch in range(NO_EPOCHS):
        unet.train()
        train_loss = []
        train_iou = []
        print(epoch)
        progress_bar = ChargingBar('Batch', max=len(train_loader))

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            target = data["masks"].to(device)
            out = get_output(data, unet)
            loss = dice_loss(out, target)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

            with torch.no_grad():
                train_loss.append(loss.item())
                train_iou.append(iou(out.argmax(dim=1), target.argmax(dim=1), 8, ignore=7, per_image=True))
                progress_bar.next()

        progress_bar.finish()

        torch.save({
            'epoch': epoch,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': lr_scheduler.state_dict()
        },
            "checkpoint_run5.pth"
        )

        wandb.save('*.pth')

        validation_imgs = []
        loss_final = []
        iou_final = []
        unet.eval()

        if epoch % 5 == 0:
            with torch.no_grad():
                for i, data in enumerate(validation_loader):
                    target = data["masks"].to(device)
                    out = get_output(data, unet)
                    loss = dice_loss(out, target)
                    loss_final.append(loss.item())
                    iou_final.append(iou(out.argmax(dim=1), target.argmax(dim=1), 8, ignore=7, per_image=True))
                    if i < len(validation_loader) / 2:
                        validation_imgs += validation_images(data, out)

            mean_train_loss = mean(train_loss)
            mean_train_iou = np.mean(train_iou)

            mean_val_loss = mean(loss_final)
            mean_val_iou = np.mean(iou_final)

            # lr_scheduler.step(mean_val_iou)
            # lr_step = optimizer.state_dict()["param_groups"][0]["lr"]

            wandb.log({"Train IOU": mean_train_iou, "Train Loss": mean_train_loss, "Validation IOU": mean_val_iou,
                       "Validation Loss": mean_val_loss, "validation": validation_imgs})

    torch.save(unet.state_dict(), WEIGHTS_FILE)
    wandb.save(WEIGHTS_FILE)