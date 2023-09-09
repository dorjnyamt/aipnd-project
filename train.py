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


def train_model(
    savedir,
    model,
    criterion,
    optimizer,
    scheduler,
    epochs,
    dataloaders,
    dataset_sizes,
    device,
):
    # Create a temporary directory to save training checkpoints
    best_model_params_path = os.path.join(savedir, "model.pth")

    # torch.save(model.state_dict(), best_model_params_path)
    utils.save_model(epochs, model, optimizer, criterion)
    best_acc = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a network")
    parser.add_argument("data_directory")
    parser.add_argument(
        "--save_dir", type=str, default="out", help="Set directory to save checkpoints"
    )
    parser.add_argument("--arch", type=str, default="vgg16", help="Choose architecture")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--hidden_units", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpu", action="store_true")

    args = parser.parse_args()
    print(vars(args))

    os.makedirs(args.save_dir, exist_ok=True)
    root_dir = args.data_directory
    image_datasets = image_datasets(root_dir)
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid", "test"]}
    dataloaders = image_loaders(image_datasets)
    print(image_datasets)
    n_features = len(image_datasets["train"].classes)

    config = ""
    if args.arch == "vgg16":
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(
            in_features=model.classifier[6].in_features, out_features=n_features
        )
    elif args.arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(in_features=model.fc_in_features, out_features=n_features)
    else:
        raise ValueError(f"Model architecture {args.arch} is not supported")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(
        args.save_dir,
        model,
        criterion,
        optimizer,
        exp_lr_scheduler,
        args.epochs,
        dataloaders,
        dataset_sizes,
        device,
    )
