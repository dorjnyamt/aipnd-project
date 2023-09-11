# Imports here
import argparse
import json

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from PIL import Image


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image_path) as image:
        image.thumbnail((256, 256))
        width, height = image.size
        left = (width - 224)/2
        top = (height - 224)/2
        right = (width + 224)/2
        bottom = (height + 224)/2
        image = image.crop((left, top, right, bottom))
        np_img = np.array(image) / 255
        np_img = (np_img - np.array([0.485, 0.456, 0.406])
                  ) / np.array([0.229, 0.224, 0.225])
        np_img = np_img.transpose((2, 0, 1))
        return np_img


def predict(image_path, model, topk=5):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.Tensor)
    image_tensor = image_tensor.to(device)
    model_input = image_tensor.unsqueeze(0)
    probs = torch.exp(model(model_input))
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().cpu().numpy().tolist()[0]
    top_labs = top_labs.detach().cpu().numpy().tolist()[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[lab] for lab in top_labs]
    return top_probs, classes

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16()
        model.classifier = nn.Sequential(
            nn.Linear(
                in_features=model.classifier[0].in_features,
                out_features=checkpoint['hidden_units'],
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=checkpoint['hidden_units'],
                      out_features=checkpoint['hidden_units']),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(
                in_features=checkpoint['hidden_units'],
                out_features=checkpoint['n_features'],
            ),
            nn.LogSoftmax(dim=1),
        )
    elif checkpoint['arch'] == 'resnet18':
        model = models.resnet18()
        model.fc = nn.Sequential(
            nn.Linear(
                in_features=model.fc.in_features,
                out_features=checkpoint['hidden_units'],
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=checkpoint['hidden_units'],
                      out_features=checkpoint['hidden_units']),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(
                in_features=checkpoint['hidden_units'],
                out_features=checkpoint['n_features'],
            ),
            nn.LogSoftmax(dim=1),
        )
    else:
        raise ValueError(f"Model architecture is not supported")
        
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict flower name from an image')
    parser.add_argument('image_path')
    parser.add_argument('chkpt_path')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of top predicted classes')
    parser.add_argument('--category_names', type=str)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    print(vars(args))
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    loaded_model = load_checkpoint(args.chkpt_path)
    loaded_model.to(device)
    probs, classes = predict(args.image_path, loaded_model, args.top_k)
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            names = [cat_to_name[cat] for cat in classes]
            print(names)
            print(probs)
            
    else:
        print(classes)
        print(probs)
        
