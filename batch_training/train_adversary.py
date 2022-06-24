import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import time
import tqdm
from itertools import chain
from torch.backends import cudnn
import pickle
import argparse
from defenses import *
from utils import *
import os

cudnn.benchmark = True




############################## DATASET AND TRAINING CODE ##############################

def train_with_distillation(model, loader, test_loader, optimizer, scheduler, loss_fn, num_epochs=50, temperature=3, oracle_defense=None, print_every=100,
                            save_every_epoch=False, save_path=None, use_argmax_countermeasure=False):
    """
    Trains the provided model on the distillation dataset
    
    :param model: the student model to train with distillation
    :param loader: the data loader for distillation; the dataset was created with make_distillation_dataset
    :param loss_fn: the loss function to use for distillation
    :param num_epochs: the number of epochs to train for
    :param oracle_defense: if true, use an online perturbation; constitutes an oracular defense method
    """
    
    for epoch in range(num_epochs):
        if (save_every_epoch == True) and (save_path != None):
            model.eval()
            torch.save(model.state_dict(), args.save_path.split('.pt')[0] + f'_{epoch}epochs.pt')
            model.train()

        with torch.no_grad():
            loss, acc = evaluate(model, test_loader)
        print('Epoch: {}, Test Loss: {:.3f}, Test Acc: {:.3f}'.format(epoch, loss, acc))
        for i, (tmp1, distill_targets) in enumerate(loader):
            bx, by = tmp1
            bx = bx.cuda()
            by = by.cuda()
            distill_targets = distill_targets.cuda()
            if use_argmax_countermeasure:
                tmp = torch.zeros_like(distill_targets)
                tmp[range(len(distill_targets)), distill_targets.argmax(dim=1)] = 1
                distill_targets = tmp
            
            if oracle_defense is not None:
                teacher, epsilons, override_grad, watermark, backprop_modules = oracle_defense
                model.eval()
                if watermark is not False:  # in watermarking experiments, get the watermark grad here and pass it in as an override
                    watermark_bx, watermark_by = watermark
                    override_grad = -1 * get_Gty(watermark_bx, model, watermark_by).detach()
                distill_targets = method_gradient_redirection(bx, teacher, [model], epsilons=epsilons, backprop_modules=backprop_modules,
                                                         override_grad=override_grad)[:,0,:].cuda()
                model.train()
                model.zero_grad()
                # with torch.no_grad():
                #     teacher_smax = torch.softmax(teacher(bx), dim=1)
                # print((teacher_smax - distill_targets).abs().sum(1))

            # forward pass
            logits = model(bx)
            loss = loss_fn(logits, distill_targets, by, temperature)

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

    if (save_every_epoch == True) and (save_path != None):
        model.eval()
        torch.save(model.state_dict(), args.save_path.split('.pt')[0] + f'_{num_epochs}epochs.pt')
        model.train()

    model.eval()


################################################################

def main(args):
    use_argmax_countermeasure = False
    if args.load_path.split('.pkl')[0].split('_')[-1] == 'argmax':
        args.load_path = args.load_path[:-11] + '.pkl'
        use_argmax_countermeasure = True
    
    use_vgg_countermeasure = False
    if args.load_path.split('.pkl')[0].split('_')[-1] == 'vgg':
        args.load_path = args.load_path[:-8] + '.pkl'
        use_vgg_countermeasure = True

    transfer_data, _ = load_data(args.transfer_data, train=True)
    eval_data, num_classes = load_data(args.eval_data, train=False)

    test_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.num_gpus*128, num_workers=args.num_gpus*3, shuffle=False, pin_memory=True)

    with open(args.load_path, 'rb') as f:
        perturbations_dict = pickle.load(f)
    perturbations = torch.FloatTensor(perturbations_dict[args.epsilon])
    print(len(transfer_data), len(perturbations))
    transfer_data = make_distillation_dataset(transfer_data, perturbations)

    if args.loss == 'cross-entropy':
        loss = cross_entropy_loss
    elif args.loss == 'distillation':
        loss = distillation_loss_clf

    ################# TRAINING #################
    print('\nTraining model on: {}\nwith transfer_data: {}\nwith eval_data: {}\n'.format(args.load_path, args.transfer_data, args.eval_data))

    student = load_model(args.eval_data, num_classes, use_vgg_countermeasure=use_vgg_countermeasure)
    num_epochs = 50
    lr = 0.01 if args.eval_data == 'cub200' else 0.1
    lr = 0.01 if use_vgg_countermeasure else lr  # assumes CIFAR data
    temperature = args.distillation_temperature
    loader = torch.utils.data.DataLoader(transfer_data, shuffle=True, pin_memory=True,
                                        batch_size=args.num_gpus*128, num_workers=args.num_gpus*3)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(loader))

    if args.oracle != 'None':
        teacher = load_model(args.eval_data, num_classes)
        teacher_path = 'your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/{}_teacher.pt'.format(args.eval_data)
        assert os.path.exists(teacher_path), 'Expected model in teacher path: {}'.format(teacher_path)
        teacher.load_state_dict(torch.load(teacher_path))
        teacher.eval()

        num_params = 0
        for p in teacher.parameters():  # used for override_grad; assuming teacher and student have same architecture
            num_params += p.numel()

        override_grad = False
        watermark = False
        if args.oracle in ['ALL-ONES', 'ALL-ONES_focused']:
            override_grad = -1 * torch.ones(num_params).cuda()
        elif args.oracle in ['WATERMARK']:
            watermark_seed = int(args.save_path.split('_')[-1].split('.')[0])
            watermark_bx, watermark_by = get_watermark_batch(eval_data, num_classes, num_watermarks=1, seed=watermark_seed)
            watermark = (watermark_bx, watermark_by)
        
        backprop_modules = [student.module.conv1] if args.oracle in ['MIN-IP_focused', 'ALL-ONES_focused'] else None
        oracle_args = (teacher, [float(args.epsilon)], override_grad, watermark, backprop_modules)
    else:
        oracle_args = None

    train_with_distillation(student, loader, test_loader, optimizer, scheduler, loss,
                            num_epochs=num_epochs, temperature=temperature, oracle_defense=oracle_args,
                            save_every_epoch=args.save_every_epoch, save_path=args.save_path,
                            use_argmax_countermeasure=use_argmax_countermeasure)

    if args.save_every_epoch == False:
        print('\n\nDone! Saving model to: {}\n'.format(args.save_path))
        torch.save(student.state_dict(), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get queryset used by the adversary.')
    parser.add_argument('--transfer_data', type=str, help='dataset that we transfer knowledge with')
    parser.add_argument('--eval_data', type=str, help='dataset that we eval with; the teacher was trained on this')
    parser.add_argument('--load_path', type=str, help='path for loading perturbed posteriors (at multiple epsilons)')
    parser.add_argument('--oracle', type=str, help='whether to use oracle', choices=['None', 'MIN-IP', 'ALL-ONES', 'MIN-IP_focused', 'ALL-ONES_focused', 'WATERMARK'],
                        default='None')
    parser.add_argument('--epsilon', type=str, help='epsilon of perturbations to use for training')
    parser.add_argument('--loss', type=str, help='loss to use for training', choices=['cross-entropy', 'distillation'], default='distillation')
    parser.add_argument('--distillation_temperature', type=float, help='temperature to use for distillation', default=1)
    parser.add_argument('--save_path', type=str, help='path for saving model')
    parser.add_argument('--num_gpus', type=int, help='number of GPUs for training', default=1)
    parser.add_argument('--save_every_epoch', action='store_true', help='saves a model at every intermediate epoch')

    args = parser.parse_args()
    print(args)

    main(args)
