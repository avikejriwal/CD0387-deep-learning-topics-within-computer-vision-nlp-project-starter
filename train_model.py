#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, criterion):
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, criterion, optimizer, *, epochs=2, device='cpu'):
    
    model=model.to(device)
    model.train()
    hook.set_mode(smd.modes.TRAIN)
    
    for e in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_idx = batch_idx.to(device)
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        e,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    
    return model


def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model


def create_data_loaders(args):
    
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resizing = (224, 224)
    
    training_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(resizing),
            transforms.ToTensor(),
            normalizer
        ]
    )
    
    testing_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(resizing),
            normalizer
        ]
    )
    
    data_source = os.environ["SM_CHANNEL_TRAINING"]
    
    train_source = os.path.join(data_source, 'train')
    test_source = os.path.join(data_source, 'valid')
    
    train_data = torchvision.datasets.ImageFolder(
        root=train_source, transform=training_transform
    )
    
    test_data = torchvision.datasets.ImageFolder(
        root=test_source, transform=testing_transform
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size)
    
    return train_loader, test_loader


def main(args):

    model=net()    
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    hook = get_hook(create_if_not_exists=True)
    assert hook is not None
    hook.register_hook(model)
    hook.register_loss(loss_optim)
    
    train_loader, test_loader = create_data_loaders(args)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(device)
    
    model=train(
        model, train_loader, loss_criterion,
        optimizer, epochs=args.epochs, hook=hook,
        device=device
    )
    test(model, test_loader, loss_criterion, hook=hook,
         device=device)
    
    torch.save(model, "demo-model/model.pth")


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0,
        metavar="LR", help="learning rate (default: 1.0)"
    )
    args=parser.parse_args()
    
    main(args)
