import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


def _get_cifar10_data_loaders(batch_size=64, num_workers=4, drop_last=True):
    train_set = torchvision.datasets.CIFAR10("./data/CIFAR10", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
    val_set = torchvision.datasets.CIFAR10("./data/CIFAR10", train=False, download=True,
                                           transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last)
    return train_loader, val_loader


def model_train(model: torch.nn.Module, train_loader: DataLoader, epochs: int, optimizer: optim.Optimizer, scheduler):
    """
    Trains the given torch model for the specified number of epochs

    :param model: model
    :param train_loader: Dataloader containing the training data
    :param epochs: number of training
    :param optimizer: Optimizer object for training
    :param scheduler: Learning rate scheduler
    """
    use_cuda = next(model.parameters()).is_cuda
    model.train()
    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            scheduler.step()


def train_cifar10(model: torch.nn.Module, epochs):
    """
    Trains a PyTorch model on CIFAR-10 for the specified number of epochs

    :param model: PyTorch model to train
    :param epochs: Number of epochs to train
    """
    train_loader, _ = _get_cifar10_data_loaders()
    base_lr = 0.0001
    max_lr = 0.06
    momentum = 0.9
    steps = int(len(train_loader) * epochs / 2.0)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=steps)
    model_train(model, train_loader, epochs, optimizer, scheduler)


def model_eval_torch(model: torch.nn.Module, val_loader: DataLoader):
    """
    Measures the accuracy of a PyTorch model over a given validation dataset

    :param model: model to be evaluated
    :param val_loader: Dataloader containing the validation dataset
    :return: top_1_accuracy on validation data
    """

    use_cuda = next(model.parameters()).is_cuda
    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model.eval()

    corr = 0
    total = 0
    for (i, batch) in enumerate(val_loader):
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x)
        corr += torch.sum(torch.argmax(out, dim=1) == y)
        total += x.shape[0]
    return corr / total
