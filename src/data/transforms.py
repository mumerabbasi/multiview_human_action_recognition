import torchvision.transforms as T


def get_train_transforms():
    """
    Returns the set of transformations to be applied during training.

    Includes resizing, normalization, and data augmentation
    (spatial augmentations).

    Returns:
        transform (callable): A composition of image transformations for
        training.
    """
    transform = T.Compose([
        T.Resize(size=(224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=(-10, 10)),
        # Convert PIL image to Tensor with pixel values in range [0, 1]
        T.ToTensor(),
        # Normalize using ImageNet mean and std
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform


def get_val_transforms():
    """
    Returns the set of transformations to be applied during validation/testing.

    Includes resizing and normalization, without data augmentation.

    Returns:
        transform (callable): A composition of image transformations for
        validation/testing.
    """
    transform = T.Compose([
        T.Resize(size=(224, 224)),
        # Convert PIL image to Tensor with pixel values in range [0, 1]
        T.ToTensor(),
        # Normalize using ImageNet mean and std
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform
