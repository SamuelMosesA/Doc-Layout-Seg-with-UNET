import albumentations as A

augmentations = A.Compose([
    A.RandomBrightnessContrast(p=0.7, brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3)),
    A.InvertImg(p=0.3),
    A.HueSaturationValue(p=0.5, hue_shift_limit=(-30, 30), sat_shift_limit=(-45, 45), val_shift_limit=(-20, 20))
])

class RandomRotate(object):
    """
    Random rotation

    Args:
        Range of rotation in degrees r: random rotates from -r to r
    """

    def __init__(self, rotation_range):
        self.rot_var = rotation_range

    def __call__(self, data):
        random_angle = np.random.normal(loc=0, scale=self.rot_var)
        image, masks = data["image"], data["masks"]
        image = transform.rotate(image, random_angle)
        masks = transform.rotate(masks, random_angle)

        return {"image": image, "masks": masks, "filename": data["filename"]}


  
class RandomFlip(object):
    """
    Random flip

    Args:
        flips randomly
    """

    def __call__(self, data):
        random_var = np.random.randint(2);
        image, masks = data["image"], data["masks"]

        if random_var == 1:
            image = np.fliplr(image).copy()
            masks = np.fliplr(masks).copy()

        return {"image": image, "masks": masks, "filename": data["filename"]}



class toTensor(object):
    """
        Converts image masks to Pytorch Tensor and calculates distance map for boundary loss
    """

    def __init__(self, val=False):
        self.img_transform = T.Compose([
            T.Normalize((0.807364995678337, 0.8255694485876925, 0.8081173673702616,0.5,0.5),
                        (0.20619193505040878, 0.21304207788633844, 0.2254893851228971,0.2886729,0.2886741))
        ])
        self.val = val

    def __call__(self, data):
        image, masks = data["image"], data["masks"]
        if not self.val:
            transformed = augmentations(image = image[:,:,:3])
            image[:,:,:3] = transformed["image"]

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        image[:3]= image[:3]/255
        image[3] = image[3]/256
        image[4] = image[4]/384
        image = self.img_transform(image)

        masks = masks.transpose((2, 0, 1))
        masks = torch.FloatTensor(masks)

        return {"image": image, "masks": masks, "filename": data["filename"]}
