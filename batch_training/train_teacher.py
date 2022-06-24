import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import tqdm
from torch.backends import cudnn
import pickle
import argparse
from defenses import *
from utils import *

cudnn.benchmark = True




############################## DATASET AND TRAINING CODE ##################################

def train(model, loader, test_loader, optimizer, scheduler, loss_fn, num_epochs=50, print_every=100):
    """
    Trains the provided model
    
    :param model: the student model to train with distillation
    :param loader: the data loader for distillation; the dataset was created with make_distillation_dataset
    :param loss_fn: the loss function to use for distillation
    :param num_epochs: the number of epochs to train for
    """
    
    for epoch in range(num_epochs):
        with torch.no_grad():
            loss, acc = evaluate(model, test_loader)
        print('Epoch: {}, Test Loss: {:.3f}, Test Acc: {:.3f}'.format(epoch, loss, acc))
        for i, (bx, by) in enumerate(loader):
            bx = bx.cuda()
            by = by.cuda()

            # forward pass
            logits = model(bx)
            loss = loss_fn(logits, by)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % print_every == 0:
                print(i, loss.item())

    with torch.no_grad():
        loss, acc = evaluate(model, test_loader)
    print('Final:: Test Loss: {:.3f}, Test Acc: {:.3f}'.format(loss, acc))
    model.eval()

################################################################

def cross_entropy_loss(logits, gt_target):
    """
    :param logits: tensor of shape (N, C); the predicted logits
    :param gt_target: long tensor of shape (N,); the gt class labels
    :returns: cross entropy loss
    """
    return F.cross_entropy(logits, gt_target, reduction='mean')


def main(args):
    train_data, _ = load_data(args.dataset, train=True)
    test_data, num_classes = load_data(args.dataset, train=False)

    def misinformation_loss(logits, gt_target):
        """
        :param logits: tensor of shape (N, C); the predicted logits
        :param gt_target: long tensor of shape (N,); the gt class labels
        :returns: cross entropy loss
        """
        smax = torch.softmax(logits, dim=1)
        loss = -1 * (((1 - smax) * torch.nn.functional.one_hot(gt_target, num_classes=num_classes)).sum(1) + 1e-12).log().mean(0)
        return loss
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.num_gpus*128, num_workers=args.num_gpus*3, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.num_gpus*128, num_workers=args.num_gpus*3, shuffle=False, pin_memory=True)

    if args.misinformation:
        loss = misinformation_loss
    else:
        loss = cross_entropy_loss

    ################# TRAINING ####################
    print('\nTraining model on: {}'.format(args.dataset))

    teacher = load_model(args.dataset, num_classes)
    num_epochs = 50
    lr = 0.01 if args.dataset == 'cub200' else 0.1
    optimizer = torch.optim.SGD(teacher.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_loader))

    train(teacher, train_loader, test_loader, optimizer, scheduler, loss, num_epochs=num_epochs)

    print('\n\nDone! Saving model to: {}\n'.format(args.save_path))
    torch.save(teacher.state_dict(), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get queryset used by the adversary.')
    parser.add_argument('--dataset', type=str, help='dataset that we transfer knowledge with')
    parser.add_argument('--save_path', type=str, help='path for saving model')
    parser.add_argument('--num_gpus', type=int, help='number of GPUs for training', default=1)
    parser.add_argument('--misinformation', type=str,
        help='if "1", train a network with the misinformation loss for the Adaptive Misinformation method', default='0')

    args = parser.parse_args()
    print(args)

    main(args)
