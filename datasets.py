import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
# batch size
BATCH_SIZE = 64


def image_datasets(root_dir):
    train_dir = root_dir + '/train'
    valid_dir = root_dir + '/valid'
    test_dir = root_dir + '/test'
    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_dir, transform=image_transforms['valid']),
        'test': datasets.ImageFolder(root=test_dir, transform=image_transforms['test'])
    }
    return image_datasets


def image_loaders(dataset):
    dataloaders = {
        'train': DataLoader(dataset=dataset['train'], batch_size=64, shuffle=True),
        'valid': DataLoader(dataset=dataset['valid'], batch_size=64, shuffle=True),
        'test': DataLoader(dataset=dataset['test'], batch_size=64, shuffle=True)
    }
    return dataloaders


image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=45),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]),
}
