import torch
from torch import nn, optim
import numpy as np
import time
import tqdm
from itertools import chain
from torch.backends import cudnn
import argparse
from defenses import *
from utils import *
import pickle
import os
from itertools import chain

cudnn.benchmark = True


############################### CIFAR EXPERIMENTS #################################

def main(args):
    assert not (args.defense == 'None' and len(args.epsilons.split()) != 1), 'set --epsilons 0 with --defense None'

    # if eval_perturbations flag is True, swap transfer dataset for eval dataset's test split,
    # but don't change args.transfer_data (b/c some defenses use surrogates specific to the transfer set)
    transfer_data, _ = load_data(args.eval_data if args.eval_perturbations else args.transfer_data,
                                 train=not args.eval_perturbations, deterministic=True)
    eval_data, num_classes = load_data(args.eval_data, train=False)
    
    teacher = load_model(args.eval_data, num_classes)
    teacher_path = 'your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/{}_teacher.pt'.format(args.eval_data.split('_')[0])
    assert os.path.exists(teacher_path), 'Expected model in teacher path: {}'.format(teacher_path)
    teacher.load_state_dict(torch.load(teacher_path))
    teacher.eval()

    num_params = 0
    for p in teacher.parameters():  # used for override_grad; assuming teacher and student have same architecture
        num_params += p.numel()

    print('\nRunning defense: {}\nwith epsilons: {}'.format(args.defense, args.epsilons))
    epsilons = [float(eps) for eps in args.epsilons.split()]
    if args.defense == 'MAD':
        surrogate = load_model(args.eval_data, num_classes)
        surr_root = 'your-path-here/model-stealing-defenses/batch_training/outputs/trained_models'
        surrogate.load_state_dict(torch.load('{}/{}_to_{}_surrogate_{}epochs.pt'.format(surr_root, args.transfer_data, args.eval_data, 0)))
        surrogate.eval()

        if args.eval_data in ['cifar10', 'cifar100', 'cub200']:
            backprop_modules = [surrogate.module.fc]  # only target the final layer, per the official MAD GitHub repository
        elif args.eval_data in ['mnist', 'fashionmnist']:
            backprop_modules = [surrogate.module.f5]
        perturbations = generate_perturbations(transfer_data, teacher, surrogate,
                                               method_orekondy, epsilons=epsilons, batch_size=4,
                                               backprop_modules=backprop_modules)
    elif args.defense in ['MIN-IP', 'ALL-ONES', 'MIN-IP_focused', 'ALL-ONES_focused', 'WATERMARK', 'MAX-NORM', 'MAX-NORM_focused',
                          'ALL-ONES_0', 'ALL-ONES_10', 'ALL-ONES_20', 'ALL-ONES_30', 'ALL-ONES_40',
                          'ALL-ONES_0_s1', 'ALL-ONES_10_s1', 'ALL-ONES_20_s1', 'ALL-ONES_30_s1', 'ALL-ONES_40_s1',
                          'ALL-ONES_10_focused1', 'ALL-ONES_10_focused2', 'ALL-ONES_10_focused3', 'ALL-ONES_10_focused4', 'ALL-ONES_10_focused5',
                          'MisinformationDirection_10', 'HelpfulDirection_10',
                          'RandomDirection1_0', 'RandomDirection1_10', 'RandomDirection1_20', 'RandomDirection1_30', 'RandomDirection1_40', 'RandomDirection1_50',
                          'RandomDirection2_0', 'RandomDirection2_10', 'RandomDirection2_20', 'RandomDirection2_30', 'RandomDirection2_40', 'RandomDirection2_50',
                          'RandomDirection3_0', 'RandomDirection3_10', 'RandomDirection3_20', 'RandomDirection3_30', 'RandomDirection3_40', 'RandomDirection3_50',
                          'MAX-NORM_10', 'WATERMARK_10',
                          'MIN-IP_10']:
        # ============================ LOAD SURROGATES ============================ #
        surrogates = []
        if args.defense in ['ALL-ONES_0', 'ALL-ONES_10', 'ALL-ONES_20', 'ALL-ONES_30', 'ALL-ONES_40',
                            'ALL-ONES_0_s1', 'ALL-ONES_10_s1', 'ALL-ONES_20_s1', 'ALL-ONES_30_s1', 'ALL-ONES_40_s1',
                            'ALL-ONES_10_focused1', 'ALL-ONES_10_focused2', 'ALL-ONES_10_focused3', 'ALL-ONES_10_focused4', 'ALL-ONES_10_focused5',
                            'MisinformationDirection_10', 'HelpfulDirection_10',
                            'RandomDirection1_0', 'RandomDirection1_10', 'RandomDirection1_20', 'RandomDirection1_30', 'RandomDirection1_40', 'RandomDirection1_50',
                            'RandomDirection2_0', 'RandomDirection2_10', 'RandomDirection2_20', 'RandomDirection2_30', 'RandomDirection2_40', 'RandomDirection2_50',
                            'RandomDirection3_0', 'RandomDirection3_10', 'RandomDirection3_20', 'RandomDirection3_30', 'RandomDirection3_40', 'RandomDirection3_50',
                            'MAX-NORM_10', 'WATERMARK_10',
                            'MIN-IP_10']:
            surrogate = load_model(args.eval_data, num_classes)
            surr_root = 'your-path-here/model-stealing-defenses/batch_training/outputs/trained_models'
            tmp_transfer_data = args.eval_data if (len(args.defense.split('_')) == 3 and args.defense.split('_')[2] == 's1') else args.transfer_data
            surrogate_path = '{}/{}_to_{}_surrogate_{}epochs.pt'.format(surr_root, tmp_transfer_data, args.eval_data, args.defense.split('_')[1])
            surrogate.load_state_dict(torch.load(surrogate_path))
            print('Loaded surrogate from', surrogate_path)
            surrogate.eval()
            surrogates.append(surrogate)
        else:
            for i in [1, 2, 3, 4]:
                surrogate = load_model(args.eval_data, num_classes)
                surr_root = 'your-path-here/model-stealing-defenses/batch_training/outputs/trained_models'
                surrogate.load_state_dict(torch.load('{}/{}_to_{}_surrogate_{}epochs.pt'.format(surr_root, args.transfer_data, args.eval_data, i*10)))
                surrogate.eval()
                surrogates.append(surrogate)

        # ============================ SELECT BACKPROP_MODULES ============================ #
        # backprop_modules is used to specify which parameters to target in the GRAD^2 or MAD defenses
        # (NOTE: this are not used in the GRAD^2 defense from the paper and is included for completeness)
        if args.defense in ['MIN-IP_focused', 'ALL-ONES_focused', 'MAX-NORM_focused']:
            backprop_modules = [[surrogate.module.fc] for surrogate in surrogates]
        elif args.defense in ['ALL-ONES_10_focused1', 'ALL-ONES_10_focused2', 'ALL-ONES_10_focused3', 'ALL-ONES_10_focused4', 'ALL-ONES_10_focused5']:
            module_id = args.defense.split('_')[2]
            if module_id == 'focused1':
                backprop_modules = [[surrogate.module.conv1] for surrogate in surrogates]
            elif module_id == 'focused2':
                backprop_modules = [[surrogate.module.block1] for surrogate in surrogates]
            elif module_id == 'focused3':
                backprop_modules = [[surrogate.module.block2] for surrogate in surrogates]
            elif module_id == 'focused4':
                backprop_modules = [[surrogate.module.block3] for surrogate in surrogates]
            elif module_id == 'focused5':
                backprop_modules = [[surrogate.module.fc] for surrogate in surrogates]
            else: raise NotImplementedError
        else:
            backprop_modules = [None for _ in surrogates]

        # ============================ SELECT OVERRIDE_GRAD ============================ #
        # override_grad is used to specify a fixed target direction for the gradient redirection. By default, the gradient_redirection function
        # uses the negative parameters of the student network as a target (min-inner-product, or MIN-IP); this overrides that functionality.
        # This is how the ALL-ONES and watermark experiments from the paper are specified; there are some additional settings as well
        # that didn't make it into the paper.
        if args.defense in ['ALL-ONES', 'ALL-ONES_focused', 'ALL-ONES_0', 'ALL-ONES_10', 'ALL-ONES_20', 'ALL-ONES_30', 'ALL-ONES_40',
                            'ALL-ONES_0_s1', 'ALL-ONES_10_s1', 'ALL-ONES_20_s1', 'ALL-ONES_30_s1', 'ALL-ONES_40_s1',
                            'ALL-ONES_10_focused1', 'ALL-ONES_10_focused2', 'ALL-ONES_10_focused3', 'ALL-ONES_10_focused4', 'ALL-ONES_10_focused5']:
            if args.defense in ['ALL-ONES', 'ALL-ONES_0', 'ALL-ONES_10', 'ALL-ONES_20', 'ALL-ONES_30', 'ALL-ONES_40',
                                'ALL-ONES_0_s1', 'ALL-ONES_10_s1', 'ALL-ONES_20_s1', 'ALL-ONES_30_s1', 'ALL-ONES_40_s1']:
                tmp_num_params = num_params
            elif args.defense in ['ALL-ONES_focused',
                                  'ALL-ONES_10_focused1', 'ALL-ONES_10_focused2', 'ALL-ONES_10_focused3', 'ALL-ONES_10_focused4', 'ALL-ONES_10_focused5']:
                tmp_num_params = 0
                for gen in [x.parameters() for x in backprop_modules[0]]:  # used for override_grad; assuming teacher and student have same architecture
                    for p in gen:
                        tmp_num_params += p.numel()
            override_grad = [-1 * torch.ones(tmp_num_params).cuda() for i in range(len(surrogates))]  # b/c we are using sample_surrogates
        elif args.defense in ['MisinformationDirection_10']:
            misinformation_model = load_model(args.eval_data, num_classes)
            misinformation_model_path = 'your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/{}_misinformation.pt'.format(args.eval_data.split('_')[0])
            assert os.path.exists(misinformation_model_path), 'Expected model in misinformation_model path: {}'.format(misinformation_model_path)
            misinformation_model.load_state_dict(torch.load(misinformation_model_path))
            misinformation_model.eval()

            misinformation_params = []
            for p in misinformation_model.parameters():
                misinformation_params.append(p.view(-1).data)
            misinformation_params = torch.cat(misinformation_params)

            override_grad = [-1 * misinformation_params]  # singleton b/c we are currently only using the _10 surrogate for this method
        elif args.defense in ['HelpfulDirection_10']:
            teacher_params = []
            for p in teacher.parameters():
                teacher_params.append(p.view(-1).data)
            teacher_params = torch.cat(teacher_params)

            override_grad = [-1 * teacher_params]  # singleton b/c we are currently only using the _10 surrogate for this method
        elif args.defense in ['RandomDirection1_0', 'RandomDirection1_10', 'RandomDirection1_20', 'RandomDirection1_30', 'RandomDirection1_40', 'RandomDirection1_50',
                              'RandomDirection2_0', 'RandomDirection2_10', 'RandomDirection2_20', 'RandomDirection2_30', 'RandomDirection2_40', 'RandomDirection2_50',
                              'RandomDirection3_0', 'RandomDirection3_10', 'RandomDirection3_20', 'RandomDirection3_30', 'RandomDirection3_40', 'RandomDirection3_50']:
            seed = int(args.defense.split('_')[0][-1])
            rng = np.random.RandomState(seed)
            random_direction = torch.FloatTensor(rng.randn(num_params)).cuda()
            override_grad = [-1 * random_direction]
        elif args.defense in ['WATERMARK_10']:
            watermark_bx, watermark_by = get_watermark_batch(eval_data, num_classes, num_watermarks=1, seed=args.watermark_seed)
            all_Gty = []
            for surr in surrogates:
                all_Gty.append(get_Gty(watermark_bx, surr, watermark_by).detach())
            Gty = torch.cat(all_Gty)
            override_grad = [-1 * Gty]
        elif args.defense in ['MAX-NORM', 'MAX-NORM_focused', 'MAX-NORM_10']:
            override_grad = []
            if args.defense in ['MAX-NORM', 'MAX-NORM_10']:
                for i in range(len(surrogates)):
                    current_params = []
                    for p in surrogates[i].parameters():
                        current_params.append(p.data.detach().view(-1))
                    current_params = torch.cat(current_params, dim=0)
                    override_grad.append(-1 * current_params)
            elif args.defense == 'MAX-NORM_focused':
                for i in range(len(surrogates)):
                    current_params = []
                    for gen in [x.parameters() for x in backprop_modules[i]]:
                        for p in gen:
                            current_params.append(p.data.detach().view(-1))
                    current_params = torch.cat(current_params, dim=0)
                    override_grad.append(-1 * current_params)
        else:
            print('using MIN-IP method')
            override_grad = False

        perturbations = generate_perturbations(transfer_data, teacher, surrogates,
                                               method_gradient_redirection, epsilons=epsilons,
                                               sample_surrogates=True, override_grad=override_grad,
                                               batch_size=args.num_gpus*32, num_workers=args.num_gpus*3,
                                               backprop_modules=backprop_modules)
    elif args.defense == 'ReverseSigmoid':
        gamma = 0.1 if args.eval_data == 'cifar10' else 0.2
        perturbations = generate_perturbations(transfer_data, teacher, None, method_reverse_sigmoid, epsilons=epsilons,
                                               batch_size=args.num_gpus*64, num_workers=args.num_gpus*3, gamma=gamma)
    elif args.defense == 'AdaptiveMisinformation':
        if args.eval_data in ['cifar10', 'cifar100']:
            oe_model = load_model(args.eval_data, num_classes).module
            oe_model_path = 'your-path-here/model-stealing-defenses/outlier-exposure/CIFAR/snapshots/oe_tune/{}_wrn_oe_tune_epoch_9.pt'.format(args.eval_data.split('_')[0])
            assert os.path.exists(oe_model_path), 'Expected model in oe_model path: {}'.format(oe_model_path)
            oe_model.load_state_dict(torch.load(oe_model_path))
            oe_model.eval().cuda()
        else:
            oe_model = None  # this means that we just use the teacher's MSP
        
        misinformation_model = load_model(args.eval_data, num_classes)
        misinformation_model_path = 'your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/{}_misinformation.pt'.format(args.eval_data.split('_')[0])
        assert os.path.exists(misinformation_model_path), 'Expected model in misinformation_model path: {}'.format(misinformation_model_path)
        misinformation_model.load_state_dict(torch.load(misinformation_model_path))
        misinformation_model.eval()

        perturbations = generate_perturbations(transfer_data, teacher, None, method_adaptive_misinformation, epsilons=epsilons,
                                               batch_size=args.num_gpus*64, num_workers=args.num_gpus*3, oe_model=oe_model, misinformation_model=misinformation_model)
    elif args.defense == 'None':
        perturbations = generate_perturbations(transfer_data, teacher, None, method_no_perturbation, epsilons=None,
                                               batch_size=args.num_gpus*64, num_workers=args.num_gpus*3)
    elif args.defense == 'Random':
        perturbations = generate_perturbations(transfer_data, teacher, None, method_rand_perturbation, epsilons=epsilons,
                                               batch_size=args.num_gpus*64, num_workers=args.num_gpus*3, num_classes=num_classes)

    perturbations_dict = {}
    for i, eps in enumerate(args.epsilons.split()):
        perturbations_dict[eps] = perturbations[:, i, :].data.cpu().numpy()
    print('\nSaving outputs to: {}\n'.format(args.save_path))
    with open(args.save_path, 'wb') as f:
        pickle.dump(perturbations_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get transfer_data used by the adversary.')
    parser.add_argument('--transfer_data', type=str, help='dataset that we query the teacher on')
    parser.add_argument('--eval_data', type=str, help='dataset that we will eval the teacher/adversary on')
    parser.add_argument('--defense', type=str, help='method used by the teacher for defense')
    parser.add_argument('--epsilons', type=str, help='epsilons to use for perturbations', default='')
    parser.add_argument('--save_path', type=str, help='path for saving perturbed posteriors')
    parser.add_argument('--num_gpus', type=int, help='number of GPUs for generating perturbations', default=1)
    parser.add_argument('--eval_perturbations', action='store_true', help='if true, generate perturbations on val set of transfer dataset')
    parser.add_argument('--watermark_seed', type=int, help='random seed to use for selecting the watermark input-output pair', default=3)

    args = parser.parse_args()
    print(args)

    main(args)
