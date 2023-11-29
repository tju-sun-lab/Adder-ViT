import torch
from torchvision import datasets
from torchvision import transforms
import os


def get_loader(args):
    tr_transform = transforms.Compose([transforms.RandomCrop(args.img_size, padding=2), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.5], [0.5])])
    train = datasets.MNIST(args.data_path, train=True, download=True, transform=tr_transform)

    te_transform = transforms.Compose([transforms.Resize([args.img_size, args.img_size]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    test = datasets.MNIST(args.data_path, train=False, download=True, transform=te_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers,
                                                 drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                drop_last=False)

    return train_loader, test_loader