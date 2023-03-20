import torch
import torch.utils.data
from torchvision import datasets, transforms

def get_fashion_mnist_trainsform():
    resize = transforms.Resize((224, 224))
    to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        resize,
        to_rgb,
        transforms.ToTensor(),
        normalize
    ])


def get_fashion_mnist_datasets(data_dir='./data', train=True, download=True):
    return datasets.FashionMNIST(
        data_dir,
        train=train,
        download=download,
        transform=get_fashion_mnist_trainsform()
    )

def get_fashion_mnist_loaders(train_dataset, test_dataset, batch_size=32, workers=4):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # must be False
        num_workers=workers,
        pin_memory=True
    )

    return train_loader, test_loader

def get_fashion_mnist_labels():
    return [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot',
    ]
