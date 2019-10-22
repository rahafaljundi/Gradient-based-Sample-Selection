# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import numpy as np
import quadprog
import miosqp
import scipy as sp
import scipy.sparse as spa
from .common import MLP, ResNet18

miosqp_settings = {
    # integer feasibility tolerance
    'eps_int_feas': 1e-03,
    # maximum number of iterations
    'max_iter_bb': 1000,
    # tree exploration rule
    #   [0] depth first
    #   [1] two-phase: depth first until first incumbent and then  best bound
    'tree_explor_rule': 1,
    # branching rule
    #   [0] max fractional part
    'branching_rule': 0,
    'verbose': False,
    'print_interval': 1}

osqp_settings = {'eps_abs': 1e-03,
                 'eps_rel': 1e-03,
                 'eps_prim_inf': 1e-04,
                 'verbose': False}


# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def cosine_similarity_selector_QP(x1, nb_selected=16, eps=1e-3):
    # x1=gradient memories

    x2 = None
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    G = torch.mm(x1, x2.t()) / (w1 * w2.t())  # .clamp(min=eps)
    # G=torch.mm(Gr,Gr.t())#cosine_similarity(G)

    t = G.size(0)
    G = G.double().numpy()
    G = G + np.eye(t) * eps
    # G = 0.5 * (G + G.transpose()) +np.eye(t) * eps
    # a=np.zeros(t)
    a = np.ones(t)
    C = np.ones((t, 1))
    h = np.zeros(1) + nb_selected
    # constraints for no smaller than zero coeffiecent
    C2 = np.eye(t)
    h2 = np.zeros(t)
    # constraints for no larger than one coeffiecent
    C3 = np.eye(t) * -1
    h3 = np.ones(t) * -1
    C = np.concatenate((C2, C3), axis=1)

    h = np.concatenate((h2, h3), axis=0)

    # np.concatenate((a, b), axis=0)
    # coeffiecents_np = quadprog.solve_qp(G, a, C2, h2)[0]
    # add a constraint to be at max 1
    coeffiecents_np = quadprog.solve_qp(G, a, C, h)[0]

    coeffiecents = torch.tensor(coeffiecents_np)

    _, inds = torch.sort(coeffiecents, descending=True)
    return inds


def cosine_similarity_selector_IQP_Exact(x1, solver, nb_selected, eps=1e-3, slack=0.01):
    """
    Integer programming
    """

    # x1=gradient memories

    x2 = None

    w1 = x1.norm(p=2, dim=1, keepdim=True)

    inds = torch.nonzero(torch.gt(w1, slack))[:, 0]
    print("removed due to gradients",w1.size(0)-inds.size(0))
    if inds.size(0) < nb_selected:
        print("WARNING GRADIENTS ARE TOO SMALL!!!!!!!!")
        inds = torch.arange(0, x1.size(0))
    w1 = w1[inds]
    x1 = x1[inds]
    x2 = x1 if x2 is None else x2
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    G = torch.mm(x1, x2.t()) / (w1 * w2.t())  # .clamp(min=eps)
    t = G.size(0)

    G = G.double().numpy()

    a = np.zeros(t)
    # a=np.ones(t)*-1

    # a=((w1-torch.min(w1))/(torch.max(w1)-torch.min(w1))).squeeze().double().numpy()*-0.01
    C = np.ones((t, 1))
    h = np.zeros(1) + nb_selected
    C2 = np.eye(t)

    hlower = np.zeros(t)
    hupper = np.ones(t)
    idx = np.arange(t)

    #################
    C = np.concatenate((C2, C), axis=1)
    C = np.transpose(C)
    h_final_lower = np.concatenate((hlower, h), axis=0)
    h_final_upper = np.concatenate((hupper, h), axis=0)
    #################
    G = spa.csc_matrix(G)

    C = spa.csc_matrix(C)

    solver.setup(G, a, C, h_final_lower, h_final_upper, idx, hlower, hupper, miosqp_settings, osqp_settings)
    results = solver.solve()
    print("STATUS", results.status)
    coeffiecents_np = results.x
    coeffiecents = torch.nonzero(torch.Tensor(coeffiecents_np))
    print("number of selected items is", sum(coeffiecents_np))
    if "Infeasible" in results.status:
        return inds

    return inds[coeffiecents.squeeze()]


def cosine_similarity_selector_IQP(x1, solver, nb_selected, eps=1e-3, slack=0.01):
    """
    Integer programming
    """

    # x1=gradient memories

    x2 = None

    w1 = x1.norm(p=2, dim=1, keepdim=True)

    inds = torch.nonzero(torch.gt(w1, slack))[:, 0]

    w1 = w1[inds]
    x1 = x1[inds]
    x2 = x1 if x2 is None else x2
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    G = torch.mm(x1, x2.t()) / (w1 * w2.t())  # .clamp(min=eps)
    t = G.size(0)
    # a=torch.sum(G-torch.eye(t),1)/(t-1)*-0.1
    # a=torch.max(G-torch.eye(t),1)[0]*-0.1
    # a=a.double().numpy()

    G = G.double().numpy()
    # G=G+ np.eye(t) * eps
    # G = 0.5 * (G + G.transpose()) +np.eye(t) * eps
    # a=(w1*-1).view(t).numpy()
    # a=np.zeros(t)
    a = np.ones(t) * -1

    # a=((w1-torch.min(w1))/(torch.max(w1)-torch.min(w1))).squeeze().double().numpy()*-0.01

    # h = np.zeros(1) + nb_selected
    C2 = np.eye(t)

    hlower = np.zeros(t)
    hupper = np.ones(t)
    idx = np.arange(t)
    # np.concatenate((a, b), axis=0)

    #################
    # C=np.concatenate((C2, C), axis=1)
    # C=np.transpose(C)

    # h_final_lower=np.concatenate((hlower,h), axis=0)
    # h_final_upper=np.concatenate((hupper,h), axis=0)
    #################
    G = spa.csc_matrix(G)

    C2 = spa.csc_matrix(C2)

    solver.setup(G, a, C2, hlower, hupper, idx, hlower, hupper, miosqp_settings, osqp_settings)
    results = solver.solve()
    print("STATUS", results.status)
    coeffiecents_np = results.x
    coeffiecents = torch.nonzero(torch.Tensor(coeffiecents_np))
    print("number of selected items is", sum(coeffiecents_np))
    # _,inds=torch.sort(coeffiecents,descending=True)

    return inds[coeffiecents.squeeze()]


def get_grad_vector(pp, grad_dims):
    """
     gather the gradients in one vector
    """
    grads = torch.Tensor(sum(grad_dims))
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


def add_memory_grad(pp, mem_grads, grad_dims):
    """
        This stores the gradient of a new memory and compute the dot product with the previously stored memories.
        pp: parameters

        mem_grads: gradients of previous memories
        grad_dims: list with number of parameters per layers

    """

    # gather the gradient of the new memory
    grads = get_grad_vector(pp, grad_dims)

    if mem_grads is None:

        mem_grads = grads.unsqueeze(dim=0)


    else:

        grads = grads.unsqueeze(dim=0)

        mem_grads = torch.cat((mem_grads, grads), dim=0)

    return mem_grads


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """

    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens

        self.is_cifar = ('cifar10' in args.data_file)


        if self.is_cifar:
            self.net = ResNet18(n_outputs, bias=args.bias)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.n_sampled_memories = args.n_sampled_memories
        self.n_constraints = args.n_constraints
        self.gpu = args.cuda

        self.batch_size=args.batch_size
        self.n_iter = args.n_iter


        # allocate ring buffer
        self.memory_data = torch.FloatTensor(self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(self.n_memories)
        # allocate  selected  memory
        self.sampled_memory_data = None
        self.sampled_memory_labs = None
        self.sampled_memory_taskids=None
        # allocate selected constraints
        self.constraints_data = None
        self.constraints_labs = None
        self.subselect = args.subselect  # first select from recent memory and then add to samples memories

        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())


        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0

    def forward(self, x, t=0):
        # t is there to be used by the main caller
        output = self.net(x)

        return output

    def select_random_samples(self, task):
        """
        To estimate the effectiveness of selecting random samples instead of using our strategy

        """
        #         import random
        #         seed=0
        #         torch.backends.cudnn.enabled = False
        #         torch.manual_seed(seed)
        #         np.random.seed(seed)
        #         random.seed(seed)
        inds = torch.randint(low=0, high=self.n_memories, size=(self.n_sampled_memories, 1)).squeeze()

        task = 0
        self.mem_grads = None
        self.sampled_memory_data = None
        self.sampled_memory_labs = None
        for index in inds:
            task = index / self.n_memories
            task_index = index % self.n_memories
            if not self.sampled_memory_data is None:

                self.sampled_memory_data = torch.cat(
                    (self.sampled_memory_data, self.memory_data[task][task_index].unsqueeze(0)), dim=0)
                self.sampled_memory_labs = torch.cat(
                    (self.sampled_memory_labs, self.memory_labs[task][task_index].unsqueeze(0)), dim=0)
            else:
                self.sampled_memory_data = self.memory_data[task][task_index].unsqueeze(0)
                self.sampled_memory_labs = self.memory_labs[task][task_index].unsqueeze(0)

        print("selected labels are", self.sampled_memory_labs)


    def print_taskids_stats(self,task):

        tasks=torch.unique(self.sampled_memory_taskids)
        for t in range(task+1):

            print('task number ',t,'samples in buffer',torch.eq(self.sampled_memory_taskids,t).nonzero().size(0))








    # MAIN TRAINING FUNCTION

    def observe(self, x, t, y):
        # update memory
        # temp
        # we dont use t :)

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)

        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz

        # self.select_random_samples_per_group()
        # self.select_random_samples_per_group()
        if self.sampled_memory_data is not None:
            shuffeled_inds=torch.randperm(self.sampled_memory_labs.size(0))
            effective_batch_size=min(self.n_constraints,self.sampled_memory_labs.size(0))
            b_index=0
        for iter_i in range(self.n_iter):

            # get gradients on previous constraints

            # now compute the grad on the current minibatch
            self.zero_grad()

            loss = self.ce(self.forward(x), y)
            loss.backward()
            self.opt.step()
            if self.sampled_memory_data is not None:

                     #print(random_batch_inds)
                random_batch_inds = shuffeled_inds[ b_index * effective_batch_size:b_index * effective_batch_size + effective_batch_size]
                batch_x=self.sampled_memory_data[random_batch_inds]
                batch_y = self.sampled_memory_labs[random_batch_inds]
                self.zero_grad()

                loss = self.ce(self.forward(batch_x), batch_y)
                loss.backward()
                self.opt.step()
                b_index += 1
                if b_index * effective_batch_size >= self.sampled_memory_labs.size(0):
                    b_index = 0
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0
            print("ring buffer is full, re-estimating of the constrains, we are at task", t)
            self.old_mem_grads = None
            self.cosine_sim = [1] * self.n_constraints
            self.select_samples_per_group(t)