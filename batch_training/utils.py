import torch
import numpy as np
from wrn import WideResNet
from torchvision import datasets, transforms, models
from lenet import LeNet5
from cub200_dataset import CUB200
import torch.nn.functional as F


############################## TRAINING/EVAL ##############################

def evaluate(model, loader):
    model.eval()
    running_loss = 0
    running_acc = 0
    count = 0
    
    for i, (bx, by) in enumerate(loader):
        bx = bx.cuda()
        by = by.cuda()
        logits = model(bx)
        loss = F.cross_entropy(logits, by, reduction='sum')
        
        running_loss += loss.item()
        running_acc += (logits.argmax(dim=1) == by).float().sum(0).item()
        count += by.shape[0]
    
    loss = running_loss / count
    acc = running_acc / count
    
    model.train()
    return loss, acc


def load_model(dataset_name, num_classes, drop_rate=0.3, use_vgg_countermeasure=False):
    dataset_name = dataset_name.split('_')[0]
    if dataset_name in ['cifar10', 'cifar100']:
        if use_vgg_countermeasure:
            model = models.vgg16_bn(pretrained=False, num_classes=num_classes).cuda()
        else:
            model = WideResNet(40, num_classes, widen_factor=2, dropRate=drop_rate).cuda()
        model = torch.nn.DataParallel(model)
    elif dataset_name in ['mnist', 'fashionmnist']:
        model = LeNet5().cuda()
        model = torch.nn.DataParallel(model)
    elif dataset_name in ['cub200']:
        model = models.resnet50(num_classes=1000, pretrained=True).cuda()
        model.fc = torch.nn.Linear(model.fc.weight.shape[1], 200)
        model = torch.nn.DataParallel(model)
    else:
        raise ValueError('{} is an invalid dataset!'.format(dataset))
    return model


def distillation_loss_clf(logits, distill_target, gt_target, temperature):
    """
    Takes the mean across a batch of the distillation loss described in
    "Distilling the Knowledge in a Neural Network" by Hinton et al.
    
    Divides the normalized logits of the model undergoing training by the temperature parameter.
    This way, at test time, the temperature is reverted and the model is better calibrated.
    
    :param logits: tensor of shape (N, C); the predicted logits
    :param distill_target: tensor of shape (N, C); the target posterior
    :param gt_target: long tensor of shape (N,); the gt class labels
    :param temperature: scalar; the temperature parameter
    :returns: cross entropy with temperature applied to the target logits
    """
    normalized_logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    
    # distillation loss
    target_logits = torch.clamp(distill_target, min=1e-12, max=1).log() / temperature
    target = torch.softmax(target_logits, dim=1)
    distill_loss = -1 * (normalized_logits/temperature * target).sum(1).mean(0)
    
    # # normal loss
    # normal_loss = F.cross_entropy(logits, gt_target, reduction='mean')
    
    return distill_loss# * (temperature ** 2) + normal_loss


def cross_entropy_loss(logits, distill_target, gt_target, temperature):
    """
    :param logits: tensor of shape (N, C); the predicted logits
    :param gt_target: long tensor of shape (N,); the gt class labels
    :returns: cross entropy loss
    """
    return F.cross_entropy(logits, gt_target, reduction='mean')


############################## DATA LOADING ##############################

def load_data(dataset_name, train=True, deterministic=False, seed=1):
    if train == False:
        dataset_name = dataset_name.split('_')[0]  # this might cause a bug now that we aren't splitting datasets in two
    
    if dataset_name in ['cub200']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        num_classes = 200
        dataset = CUB200(root='your-path-here/model-stealing-defenses/batch_training/condor_scripts/data',
                         train=train, transform=transform, download=False)
    
    elif dataset_name in ['caltech256']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        num_classes = 257
        dataset = datasets.ImageFolder(root='your-path-here/model-stealing-defenses/batch_training/condor_scripts/data/256_ObjectCategories',
                                      transform=transform)

    elif dataset_name in ['imagenet_cub200']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        split = 'train' if train else 'val'
        num_classes = 200
        dataset = datasets.ImageFolder(f'./data/ImageNet_CUB200/{split}', transform=transform)
        shuffle_indices = np.arange(len(dataset))
        rng = np.random.RandomState(seed)
        rng.shuffle(shuffle_indices)
        dataset = torch.utils.data.Subset(dataset, shuffle_indices[:30000])  # for comparability with Caltech256
        

    elif dataset_name in ['imagenet_cifar10', 'imagenet_cifar100']:
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
        
        split = 'train' if train else 'val'
        if dataset_name in ['imagenet_cifar10']:
            num_classes = 10
            dataset = datasets.ImageFolder(f'./data/ImageNet_CIFAR10/{split}', transform=transform)
        elif dataset_name in ['imagenet_cifar100']:
            num_classes = 100
            dataset = datasets.ImageFolder(f'./data/ImageNet_CIFAR100/{split}', transform=transform)
    
    elif dataset_name in ['cifar10', 'cifar10_1', 'cifar10_2', 'cifar100', 'cifar100_1', 'cifar100_2']:
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
        
        if dataset_name in ['cifar10', 'cifar10_1', 'cifar10_2']:
            num_classes = 10
            dataset = datasets.CIFAR10('./data', train=train, transform=transform, download=True)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)
            if dataset_name == 'cifar10_1':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[:len(shuffle_indices)//2])
            elif dataset_name == 'cifar10_2':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[len(shuffle_indices)//2:])
        elif dataset_name in ['cifar100', 'cifar100_1', 'cifar100_2']:
            num_classes = 100
            dataset = datasets.CIFAR100('./data', train=train, transform=transform, download=True)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)
            if dataset_name == 'cifar100_1':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[:len(shuffle_indices)//2])
            elif dataset_name == 'cifar100_2':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[len(shuffle_indices)//2:])
    
    elif dataset_name in ['mnist', 'mnist_1', 'mnist_2', 'fashionmnist', 'fashionmnist_1', 'fashionmnist_2']:
        if train and not deterministic:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ])
        
        if dataset_name in ['mnist', 'mnist_1', 'mnist_2']:
            num_classes = 10
            dataset = datasets.MNIST('./data', train=train, transform=transform, download=True)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)
            if dataset_name == 'mnist_1':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[:len(shuffle_indices)//2])
            elif dataset_name == 'mnist_2':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[len(shuffle_indices)//2:])
        elif dataset_name in ['fashionmnist', 'fashionmnist_1', 'fashionmnist_2']:
            num_classes = 10
            dataset = datasets.FashionMNIST('./data', train=train, transform=transform, download=True)
            shuffle_indices = np.arange(len(dataset))
            rng = np.random.RandomState(seed)
            rng.shuffle(shuffle_indices)
            if dataset_name == 'fashionmnist_1':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[:len(shuffle_indices)//2])
            elif dataset_name == 'fashionmnist_2':
                dataset = torch.utils.data.Subset(dataset, shuffle_indices[len(shuffle_indices)//2:])
    
    return dataset, num_classes


class ZippedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, distillation_targets):
            super().__init__()
            self.dataset = dataset
            self.distillation_targets = distillation_targets
            assert len(dataset) == len(distillation_targets), 'Should have same length'
    
        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return self.dataset[idx], self.distillation_targets[idx]


def make_distillation_dataset(dataset, distillation_targets):
    """
    Takes a PyTorch dataset D1 and outputs a new dataset D2, where D2[i] = (D1[i], distillation_targets[i])
    
    :param dataset: a PyTorch dataset
    :param distillation_targets: a list of distillation targets
    :returns: the new dataset
    """
    return ZippedDataset(dataset, distillation_targets)


def get_watermark_batch(test_data, num_classes, num_watermarks=1, seed=1):
    rng = np.random.RandomState(seed)

    watermark_by = torch.zeros(num_watermarks, num_classes).cuda()
    for i in range(num_watermarks):
        watermark_by[i][rng.choice(num_classes)] = 1

    watermark_bx = []
    indices = rng.choice(len(test_data), size=num_watermarks, replace=False)
    for i in range(num_watermarks):
        watermark_bx.append(test_data[indices[i]][0].cuda())
    watermark_bx = torch.stack(watermark_bx)
    
    return watermark_bx, watermark_by