# Imports
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from datasets import image_datasets, image_loaders
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune a VGG16 or ResNet18 on your own data")
    parser.add_argument("data_directory")
    parser.add_argument(
        "--save_dir", type=str, default="outputs", help="Set directory to save checkpoints"
    )
    parser.add_argument("--arch", type=str, default="vgg16",
                        help="Choose architecture")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--hidden_units", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpu", action="store_true")

    args = parser.parse_args()
    print(vars(args))

    os.makedirs(args.save_dir, exist_ok=True)
    root_dir = args.data_directory
    datasets = image_datasets(root_dir)
    dataset_sizes = {x: len(datasets[x])
                     for x in ["train", "valid", "test"]}
    dataloaders = image_loaders(datasets)
    n_features = len(datasets["train"].classes)

    # PREP MODEL
    if args.arch == "vgg16":
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(
                in_features=model.classifier[0].in_features, out_features=args.hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=args.hidden_units,
                      out_features=args.hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=args.hidden_units, out_features=n_features),
            nn.LogSoftmax(dim=1)
        )
        optimizer = optim.SGD(params=model.classifier.parameters(),
                          lr=args.learning_rate, momentum=0.9)
    elif args.arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(
                in_features=model.fc.in_features, out_features=args.hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=args.hidden_units,
                      out_features=args.hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=args.hidden_units, out_features=n_features),
            nn.LogSoftmax(dim=1)
        )
        optimizer = optim.SGD(params=model.fc.parameters(),
                          lr=args.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Model architecture {args.arch} is not supported")
    print(model)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    model = model.to(device)
    criterion = nn.NLLLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # TRAIN MODEL
    print(f'TRAINING FOR {args.epochs} EPOCHS.')
    for epoch in range(args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(
                f"Epoch {epoch + 1}/{args.epochs} {phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

    # SAVE MODEL
    model.class_to_idx = datasets['train'].class_to_idx
    savepath = os.path.join(args.save_dir, f'{args.arch}-{args.data_directory}model.pt')
    torch.save({
        'arch': args.arch,
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'hidden_units': args.hidden_units,
        'n_features': n_features
    }, savepath)

    print(f'TRAINING COMPLETE. MODEL SAVED AT {savepath}.')
