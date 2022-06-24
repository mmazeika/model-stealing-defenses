import torch
import numpy as np
from wrn import WideResNet
import tqdm
from itertools import chain
import copy


############################### PERTURBATION GENERATION UTILITIES #################################

def make_param_generator(modules):
    return chain(*[module.parameters() for module in modules])


def generate_perturbations(dataset, teacher, student, method, epsilons=[0.1], avg_posteriors=False, sample_surrogates=False,
                           batch_size=64, num_workers=4, **kwargs):
    """
    Takes a data loader, the teacher network, the student network, and the method to use for creating
    perturbations. Generates the perturbations on the dataset and returns them.
    
    :param dataset: the dataset on which to generate perturbations
    :param teacher: the teacher network with which to generate posteriors for distillation
    :param student: the student network used for generating perturbations
    :param method: function taking (bx, teacher, student) as inputs and returning the posterior perturbation
    :param epsilons: list of target epsilons
    :returns: tensor of perturbed posteriors, with shape [N, C] (N=len(dataset), C=num_classes)
    """
    shuffle_indices = np.arange(len(dataset))
    np.random.shuffle(shuffle_indices)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, shuffle_indices),
                                         batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    
    teacher_pred_perturbed = []
    for (bx, by) in tqdm.tqdm(loader, mininterval=1.0):
        bx = bx.cuda()
        if avg_posteriors:
            assert type(student) == list, 'can only use avg_posteriors=True with list of students'
            tmp_outs = []
            for stdt in student:
                out = method(bx, teacher, [stdt], epsilons=epsilons, **kwargs).cpu().detach()
                tmp_outs.append(out)
            out = sum(tmp_outs) / len(tmp_outs)
        elif sample_surrogates:
            assert type(student) == list, 'can only use sample_surrogates=True with list of students'
            sample_idx = np.random.choice(len(student))
            stdt = student[sample_idx]
            if 'backprop_modules' in kwargs:
                # if using backprop_modules with sample_surrogates, assume a list of module lists
                backprop_modules = kwargs['backprop_modules'][sample_idx]  if type(kwargs['backprop_modules']) == list else kwargs['backprop_modules']
                override_grad = kwargs['override_grad'][sample_idx] if type(kwargs['override_grad']) == list else kwargs['override_grad']
                tmp_kwargs = {k: v for (k, v) in kwargs.items() if (k != 'backprop_modules') and (k != 'override_grad')}
                tmp_kwargs['backprop_modules'] = backprop_modules
                tmp_kwargs['override_grad'] = override_grad
                out = method(bx, teacher, [stdt], epsilons=epsilons, **tmp_kwargs).cpu().detach()
            else:
                out = method(bx, teacher, [stdt], epsilons=epsilons, **kwargs).cpu().detach()
        else:
            out = method(bx, teacher, student, epsilons=epsilons,  **kwargs).cpu().detach()
        teacher_pred_perturbed.append(out)
    teacher_pred_perturbed = torch.cat(teacher_pred_perturbed, dim=0)
    
    assert len(teacher_pred_perturbed) == len(shuffle_indices), 'sanity check; can remove later'
    unshuffle_indices = np.zeros(len(dataset))
    for i, p in enumerate(shuffle_indices):
        unshuffle_indices[p] = i
    teacher_pred_perturbed = teacher_pred_perturbed[unshuffle_indices]
    
    return teacher_pred_perturbed


############################### GRADIENT FUNCTIONS #################################

def inner_product(x1, x2):
    return (x1 * x2).sum()

def cosine(x1, x2):
    norm1 = x1.view(-1).norm(p=2)
    norm2 = x2.view(-1).norm(p=2)
    return (x1 * x2).sum() / (norm1 * norm2)


def get_Gty(bx, student, y, create_graph=False):
    """
    computes G^T y, where y is the posterior used in cross-entropy loss
    
    Works with student_pred of shape (N, K)
    
    :param bx: a batch of image inputs (CIFAR or ImageNet in our experiments)
    :param student: student network
    :param y: posterior to be used for cross-entropy loss
    :create_graph: whether to create the graph for double backprop
    :returns: G^T y, a 1D tensor with length equal to the number of params in backprop_modules
    """
    logits = student(bx)
    normalized_logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # cross-entropy loss
    loss = (y * normalized_logits).sum(1).mean(0)
    student.zero_grad()
    student_grad = torch.autograd.grad([loss],
                                       student.parameters(),
                                       create_graph=create_graph,
                                       retain_graph=False,
                                       only_inputs=True)
    Gty = torch.cat([x.view(-1) for x in student_grad])
    
    return Gty


def make_param_generator(modules):
    """
    :param modules: a list of PyTorch modules
    :returns: a generator that chains together all the individual parameter generators
    """
    return chain(*[module.parameters() for module in modules])

def get_Gty_partial(bx, student, y, backprop_modules, create_graph=False):
    """
    Same as get_Gty, but only returns gradient wrt the provided parameters specified in backprop_modules.
    
    This is a component of the MAD defense specified in the official MAD GitHub repository. It speeds up
    their method considerably by only backpropping through part of the network.
    We didn't explore this functionality with GRAD^2, since it is already fast enough,
    but it could be used to make things even faster.
    
    :param bx: a batch of image inputs (CIFAR or ImageNet in our experiments)
    :param student: student network
    :param y: posterior to be used for cross-entropy loss
    :backprop_modules: modules of the student network to backprop through
    :create_graph: whether to create the graph for double backprop
    :returns: G^T y, a 1D tensor with length equal to the number of params in backprop_modules
    """
    for p in student.parameters():
        p.requires_grad = False
    for p in make_param_generator(backprop_modules):
        p.requires_grad = True
    
    logits = student(bx)
    normalized_logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # cross-entropy loss
    loss = (y * normalized_logits).sum(1).mean(0)
    student.zero_grad()
    student_grad = torch.autograd.grad([loss],
                                       make_param_generator(backprop_modules),
                                       create_graph=create_graph,
                                       retain_graph=False,
                                       only_inputs=True)
    Gty = torch.cat([x.view(-1) for x in student_grad])
    
    for p in student.parameters():
        p.requires_grad = True
    
    return Gty




############################### DEFENSE METHODS #################################


# num_params = 0
# for p in model.parameters():
#     num_params += p.numel()
# random_vector = np.random.choice([-1,1], size=num_params)

def method_adaptive_misinformation(bx, teacher, student=None, epsilons=[0.1], oe_model=None, misinformation_model=None):
    """
    :param bx: input batch of tensors for teacher and student networks
    :param teacher: teacher network, outputs logits
    :param student: list of students (None for reverse sigmoid)
    :param epsilons: a list of tau parameters in the adaptive misinformation method (named epsilon for convenience)
    :param oe_model: the outlier exposure model to use for detecting OOD data; if None, then use the MSP of the teacher
    :param misinformation_model: the misinformation model in the adaptive misinformation defense
    :returns: perturbed posterior
    """
    assert misinformation_model is not None, 'must supply a misinformation model'

    taus = epsilons  # renaming for clarity
    nu = 1000

    # get clean posterior
    with torch.no_grad():
        teacher_logits = teacher(bx)
        teacher_pred = torch.softmax(teacher_logits, dim=1)
    
    # get misinformation posterior
    with torch.no_grad():
        misinformation_logits = misinformation_model(bx)
        misinformation_pred = torch.softmax(misinformation_logits, dim=1)

    # get inlier scores
    if oe_model is not None:
        with torch.no_grad():
            oe_logits = oe_model(bx)
            inlier_scores = torch.softmax(oe_logits, dim=1).max(dim=1)[0]
    else:
        inlier_scores = teacher_pred.max(dim=1)[0]
    
    all_outs = []
    # for each beta, perturb the posterior with the adaptive misinformation method
    for tau in taus:
        alpha = torch.reciprocal(1 + torch.exp(nu * (inlier_scores - tau))).unsqueeze(-1)
        out = (1 - alpha) * teacher_pred + alpha * misinformation_pred
        out = out.unsqueeze(1) # adding dimension for tau
        all_outs.append(out)
    
    all_outs = torch.cat(all_outs, dim=1)

    return all_outs.detach()


def method_reverse_sigmoid(bx, teacher, student=None, epsilons=[0.1], gamma=0.1):
    """
    :param bx: input batch of tensors for teacher and student networks
    :param teacher: teacher network, outputs logits
    :param student: list of students (None for reverse sigmoid)
    :param epsilons: a list of beta parameters in the reverse sigmoid method (named epsilon for convenience)
    :param gamma: the gamma parameter in the reverse sigmoid method
    :returns: perturbed posterior
    """
    betas = epsilons  # renaming for clarity

    # get clean posterior
    with torch.no_grad():
        teacher_logits = teacher(bx)
        teacher_pred = torch.softmax(teacher_logits, dim=1)
    
    all_outs = []
    # for each beta, perturb the posterior with the reverse sigmoid method
    for beta in betas:
        logits = torch.log(torch.clamp(teacher_pred / (1 - teacher_pred), min=1e-12, max=1-1e-12))
        z = gamma * logits
        r = beta * (torch.sigmoid(z) - 0.5)
        out = teacher_pred - r
        out = out / out.sum(-1).unsqueeze(-1)
        out = out.unsqueeze(1)  # adding dimension for beta
        all_outs.append(out)
    
    all_outs = torch.cat(all_outs, dim=1)

    return all_outs.detach()


def method_no_perturbation(bx, teacher, student=None, epsilons=None):
    """
    :param bx: input batch of tensors for teacher and student networks
    :param teacher: teacher network, outputs logits
    :returns: unperturbed posterior
    """
    with torch.no_grad():
        teacher_logits = teacher(bx)
        teacher_pred = torch.softmax(teacher_logits, dim=1)
        
    return teacher_pred.unsqueeze(1).detach()


def method_rand_perturbation(bx, teacher, student, epsilons=[0], num_classes=None):
    """
    :param bx: input batch of tensors for teacher and student networks
    :param teacher: teacher network, outputs logits
    :param student: unused
    :param epsilons: epsilons for perturbation
    :returns: random perturbation of teacher posterior satisfying L1 constraint
    """
    # for y
    teacher_logits = teacher(bx)
    teacher_pred = torch.softmax(teacher_logits, dim=1)
    
    max_indices = torch.max(teacher_pred, dim=1)[1]
    perturbation_target = torch.zeros_like(teacher_pred)
    for i in range(len(max_indices)):
        choices = list(range(num_classes))
        choices.remove(max_indices[i].int().item())
        assert len(choices) == num_classes - 1, 'error'
        loc = np.random.choice(choices)
        perturbation_target[i, loc] = 1
    
    out = []
    for epsilon in epsilons:
        tmp = (epsilon / 2) * perturbation_target + (1 - (epsilon / 2)) * teacher_pred
        out.append(tmp.unsqueeze(1).detach())
        
    return torch.cat(out, dim=1)


def method_gradient_redirection(bx, teacher, students, epsilons=[0.1], backprop_modules=None, override_grad=False):
    """
    Find perturbation minimizing inner product of perturbed gradient with original gradient.
    The perturbation is constrained to have an L1 norm of epsilon and to be such that the
    perturbed posterior is on the simplex.

    Optionally uses an ensemble of students (not used in the paper, but included for completeness)

    NOTE: The paper describes gradient redirection as maximizing the inner product, while this implementation minimizes inner product. They are functionally
    identical, but this means that to maximize inner product with the all-ones vector you should pass in override_grad=-1*torch.ones(...)

    NOTE: This function with the appropriate surrogate (i.e., student) model is the GRAD^2 defense from the paper.
    
    :param bx: input batch of tensors for teacher and student networks
    :param teacher: teacher network, outputs logits
    :param students: list of student networks
    :param epsilons: list of target epsilons (L1 distance)
    :param backprop_modules: optional list (or list of lists) of PyTorch modules from the student networks, specifying the parameters to target with the defense
                             (NOTE: This is not used in the GRAD^2 defense in the paper and is included for completeness.)
    :param override_grad: optional tensor specifying the target direction for gradient redirection; this is how the all-ones and watermark experiments are specified
    :returns: perturbation satisfying L1 constraint
    """
    # FIRST, GET DIRECTION OF PERTURBATION
    
    # get Gty_tilde and Gty for each student network
    with torch.no_grad():
        teacher_logits = teacher(bx)
        teacher_pred = torch.softmax(teacher_logits, dim=1)
    teacher_pred.requires_grad_()
    
    all_Gty_tilde = []
    all_Gty = []
    for student in students:
        student.zero_grad()
        y_tilde = teacher_pred.clone()  # initialize y_tilde to the teacher posterior
        if backprop_modules is None:
            Gty_tilde = get_Gty(bx, student, y_tilde, create_graph=True)
        else:
            Gty_tilde = get_Gty_partial(bx, student, y_tilde, backprop_modules, create_graph=True)
        Gty = Gty_tilde.detach()  # this works if we initialize y_tilde = y, as we do here
        all_Gty_tilde.append(Gty_tilde)
        all_Gty.append(Gty)
    
    Gty_tilde = torch.cat(all_Gty_tilde)
    Gty = torch.cat(all_Gty)

    if override_grad is not False:  # "is" matters here; cannot use ==
        Gty = override_grad

    # now compute the objective and double backprop
    objective = inner_product(Gty_tilde, Gty)
    grad_pred_inner_prod = torch.autograd.grad([objective], [teacher_pred], only_inputs=True)[0]
    
    # now compute the optimal perturbation using our algorithm, separately for each example in the batch
    all_teacher_pred_perturbed = []
    for idx in range(bx.shape[0]):
        teacher_pred_perturbed_per_epsilon = []
        for epsilon in epsilons:
            epsilon_target = epsilon
            
            # algorithm start
            c = torch.argsort(grad_pred_inner_prod[idx])  # smallest to largest
            take_pointer = len(c) - 1  # where to take probability mass from; for L1 constraint, we always give to c[0]

            with torch.no_grad():
                tmp = teacher_pred.clone()[idx]
                can_give = min(1 - tmp[c[0]].item(), epsilon_target/2)
                found_to_give = torch.zeros(1).cuda()[0]
                while (found_to_give < can_give):
                    found_here = tmp[c[take_pointer]].item()
                    #print('found_to_give: {} \tcan_give: {} \tfound_here: {}'.format(found_to_give, can_give, found_here))
                    if found_to_give + found_here <= can_give:
                        tmp[c[take_pointer]] -= found_here
                        found_to_give += found_here
                    elif found_to_give + found_here > can_give:
                        #print('got here')
                        tmp[c[take_pointer]] -= can_give - found_to_give
                        found_to_give += can_give - found_to_give
                    take_pointer -= 1
                    if np.isclose(found_to_give.item(), can_give):
                        break
                tmp[c[0]] += found_to_give

                # to handle arithmetic errors (very minor when they occur)
                tmp = tmp.cuda()
                teacher_pred_perturbed = torch.softmax(torch.log(torch.clamp(tmp, 1e-15, 1)), dim=0).unsqueeze(0)
            # algorithm end
            teacher_pred_perturbed_per_epsilon.append(teacher_pred_perturbed)
            
        teacher_pred_perturbed_per_epsilon = torch.cat(teacher_pred_perturbed_per_epsilon, dim=0)
        all_teacher_pred_perturbed.append(teacher_pred_perturbed_per_epsilon.unsqueeze(0))
    
    teacher_pred_perturbed = torch.cat(all_teacher_pred_perturbed, dim=0)
        
    return teacher_pred_perturbed.detach()


def method_orekondy(bx, teacher, student, epsilons=[0.1], backprop_modules=None):
    """
    Find perturbation maximizing cosine distance of perturbed gradient with original gradient
    using the method from Orekondy et al.
    
    :param bx: input batch of tensors for teacher and student networks
    :param teacher: teacher network, outputs logits
    :param student: student network, outputs logits, for backpropping through
    :param epsilons: list of target epsilons (L1 distance)
    :returns: perturbation satisfying L1 constraint
    """
    full_bx = bx
    all_perturbations = []
    for idx in range(full_bx.shape[0]):
        bx = full_bx[idx].unsqueeze(0)
    
        teacher.zero_grad()
        student.zero_grad()

        teacher_logits = teacher(bx)
        teacher_pred = torch.softmax(teacher_logits, dim=1)
        num_classes = teacher_pred.shape[1]

        y = teacher_pred.detach()
        

        if backprop_modules is None:
            Gty = get_Gty(bx, student, y, create_graph=False)
        else:
            Gty = get_Gty_partial(bx, student, y, backprop_modules, create_graph=False)

        # for each 1-hot vector, compute the objective
        max_objective = -0.1  # any negative number will do
        best_1hot = None
        for i in range(num_classes):
            y_tilde = torch.zeros(1, num_classes).cuda()
            y_tilde[0,i] = 1

            if backprop_modules is None:
                Gty_tilde = get_Gty(bx, student, y_tilde, create_graph=False)
            else:
                Gty_tilde = get_Gty_partial(bx, student, y_tilde, backprop_modules, create_graph=False)

            objective = 1 - cosine(Gty_tilde, Gty).item()
            if objective >= max_objective:
                max_objective = objective
                best_1hot = y_tilde
        
        current_perturbations_per_epsilon = []
        for epsilon in epsilons:
            alpha = epsilon / (teacher_pred - best_1hot).abs().sum()
            teacher_pred_perturbed = teacher_pred * (1 - alpha) + best_1hot * alpha
            #print((teacher_pred_perturbed - teacher_pred).abs().sum().item())
            current_perturbations_per_epsilon.append(teacher_pred_perturbed.detach())

        current_perturbations_per_epsilon = torch.cat(current_perturbations_per_epsilon, dim=0)
        all_perturbations.append(current_perturbations_per_epsilon.unsqueeze(0))
    
    all_perturbations = torch.cat(all_perturbations, dim=0)
    return all_perturbations
