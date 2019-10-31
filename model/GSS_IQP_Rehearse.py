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


# Auxiliary functions


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


def cosine_similarity_selector_IQP_Exact(x1, solver, nb_selected, eps=1e-3, slack=0.0, normalize=False, age=None,
                                         age_weight=-1):
    """
    Integer programming
    """

    x2 = None

    w1 = x1.norm(p=2, dim=1, keepdim=True)

    inds = torch.nonzero(torch.gt(w1, slack))[:, 0]
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
        self.margin = args.memory_strength
        self.is_cifar = ('cifar10' in args.data_file)
        m = miosqp.MIOSQP()
        self.solver = m
        if self.is_cifar:
            self.net = ResNet18(n_outputs, bias=args.bias)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.normalize = args.normalize
        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.n_sampled_memories = args.n_sampled_memories
        self.n_constraints = args.n_constraints
        self.gpu = args.cuda
        self.batch_size = args.batch_size
        self.n_iter = args.n_iter
        self.slack = args.slack
        self.normalize = args.normalize
        self.change_th = args.change_th  # gradient direction change threshold to re-select constraints
        # allocate ring buffer
        self.memory_data = torch.FloatTensor(self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(self.n_memories)
        # allocate  selected  memory
        self.sampled_memory_data = None
        self.sampled_memory_labs = None
        self.sampled_memory_taskids = None
        self.sampled_memory_age = None

        self.subselect = args.subselect  # if 1, first select from recent memory and then add to samples memories
        # allocate selected constraints
        self.constraints_data = None
        self.constraints_labs = None
        # old grads to measure changes
        self.old_mem_grads = None
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        # we keep few samples per task and use their gradients

        # if args.cuda:
        #    self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0


    def forward(self, x, t=0):
        # t is there to be used by the main caller
        output = self.net(x)

        return output

    def print_taskids_stats(self):

        tasks = torch.unique(self.sampled_memory_taskids)
        for t in range(tasks.size(0)):
            print('task number ', t, 'samples in buffer', torch.eq(self.sampled_memory_taskids, t).nonzero().size(0))

        for lab in torch.sort(torch.unique(self.sampled_memory_labs))[0]:
            print("number of samples from class", lab, torch.nonzero(torch.eq(self.sampled_memory_labs, lab)).size(0))

    def select_samples_per_group(self, task):
        """
        Assuming a ring buffer, backup constraints and constrains,
        re-estimate the backup constrains and constrains

        """

        print("constraints selector")
        self.mem_grads = None
        # get gradients from the ring buffer
        self.eval()
        for x, y in zip(self.memory_data, self.memory_labs):
            self.zero_grad()
            ptloss = self.ce(self.forward(x.unsqueeze(0)), y.unsqueeze(0))
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            self.mem_grads = add_memory_grad(self.parameters, self.mem_grads, self.grad_dims)

        if self.subselect:
            added_inds = cosine_similarity_selector_IQP_Exact(self.mem_grads, nb_selected=int(self.n_memories / 10),
                                                              solver=self.solver)
        else:
            added_inds = torch.arange(0, self.memory_data.size(0))
        self.print_loss(self.memory_data[added_inds], self.memory_labs[added_inds], "loss on selected samples from Mr")
        # 10 her is batch size
        print("Number of added inds from the very new batch",
              torch.ge(added_inds, self.n_memories - 10).nonzero().size(0))
        from_buffer_size = added_inds.size(0)

        new_task_ids = torch.zeros(added_inds.size(0)) + task
        new_age = torch.zeros(added_inds.size(0))
        self.new_mem_grads = self.mem_grads[added_inds].clone()
        # estimate the active constraints from the backup samples
        self.mem_grads = None
        # buffer is full
        if not self.sampled_memory_data is None and self.n_sampled_memories < (
                self.sampled_memory_data.size(0) + from_buffer_size):
            # ReDo Selection
            for x, y in zip(self.sampled_memory_data, self.sampled_memory_labs):
                self.zero_grad()
                ptloss = self.ce(self.forward(x.unsqueeze(0)), y.unsqueeze(0))
                ptloss.backward()
                # add the new grad to the memory grads and add it is cosine similarity
                self.mem_grads = add_memory_grad(self.parameters, self.mem_grads, self.grad_dims)

            # update the backup constraints:
            self.sampled_memory_data = torch.cat((self.memory_data[added_inds], self.sampled_memory_data),
                                                 dim=0).clone()
            self.sampled_memory_labs = torch.cat((self.memory_labs[added_inds], self.sampled_memory_labs),
                                                 dim=0).clone()

            self.sampled_memory_taskids = torch.cat((new_task_ids, self.sampled_memory_taskids),
                                                    dim=0).clone()
            self.sampled_memory_age = torch.cat((new_age, self.sampled_memory_age),
                                                dim=0).clone()
            self.mem_grads = torch.cat((self.new_mem_grads, self.mem_grads), dim=0)

            # select samples that minimize the feasible region
            inds = cosine_similarity_selector_IQP_Exact(self.mem_grads, nb_selected=self.n_sampled_memories,
                                                        solver=self.solver, age=self.sampled_memory_age)

            print("number of retained memories", torch.nonzero(torch.ge(inds, from_buffer_size)).size(0))
            self.print_loss(self.sampled_memory_data[inds[torch.ge(inds, from_buffer_size)]],
                            self.sampled_memory_labs[inds[torch.ge(inds, from_buffer_size)]],
                            "loss on the selected Mb Samples")
            if torch.nonzero(torch.ge(inds, from_buffer_size)).size(0) == 0:
                pdb.set_trace()
            self.sampled_memory_data = self.sampled_memory_data[inds].clone()
            self.sampled_memory_labs = self.sampled_memory_labs[inds].clone()
            self.sampled_memory_taskids = self.sampled_memory_taskids[inds].clone()
            self.sampled_memory_age = self.sampled_memory_age[inds].clone()
        else:

            if not self.sampled_memory_data is None:
                self.sampled_memory_data = torch.cat((self.memory_data[added_inds], self.sampled_memory_data),
                                                     dim=0).clone()
                self.sampled_memory_labs = torch.cat((self.memory_labs[added_inds], self.sampled_memory_labs),
                                                     dim=0).clone()
                self.sampled_memory_taskids = torch.cat((new_task_ids, self.sampled_memory_taskids),
                                                        dim=0).clone()
                self.sampled_memory_age = torch.cat((new_age, self.sampled_memory_age),
                                                    dim=0).clone()
            else:
                self.sampled_memory_data = self.memory_data[added_inds].clone()
                self.sampled_memory_labs = self.memory_labs[added_inds].clone()
                self.sampled_memory_taskids = new_task_ids.clone()
                self.sampled_memory_age = new_age.clone()
        print("selected labels are", self.sampled_memory_labs)
        self.print_taskids_stats()
        self.mem_grads = None
        self.train()


    def print_loss(self, x, y, msg):
        # estimate the loss and print it on a given batch of samples
        ptloss = self.ce(self.forward(
            x),
            y)
        print("$$", msg, "$$", ptloss)


    # MAIN TRAINING FUNCTION
    def observe(self, x, t, y):
        # update memory
        #
        # we dont use it :)

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

        if self.sampled_memory_data is not None:
            shuffeled_inds = torch.randperm(self.sampled_memory_labs.size(0))
            effective_batch_size = min(self.n_constraints, self.sampled_memory_labs.size(0))
            b_index = 0

        for iter_i in range(self.n_iter):

            # get gradients on previous constraints

            # now compute the grad on the current minibatch
            self.zero_grad()

            loss = self.ce(self.forward(x), y)
            loss.backward()
            self.opt.step()
            if self.sampled_memory_data is not None:

                random_batch_inds = shuffeled_inds[
                                    b_index * effective_batch_size:b_index * effective_batch_size + effective_batch_size]

                batch_x = self.sampled_memory_data[random_batch_inds]
                batch_y = self.sampled_memory_labs[random_batch_inds]

                self.zero_grad()

                loss = self.ce(self.forward(batch_x), batch_y)
                loss.backward()
                self.opt.step()
                b_index += 1
                if b_index * effective_batch_size >= self.sampled_memory_labs.size(0):
                    b_index = 0

        # update buffer
        if self.mem_cnt == self.n_memories:
            self.print_loss(self.memory_data, self.memory_labs, msg="Mr Loss Before Buffer rehearsal")

            if self.sampled_memory_labs is not None:
                self.print_loss(self.memory_data, self.memory_labs, msg="Mr Loss Before selection")
                self.print_loss(self.sampled_memory_data, self.sampled_memory_labs, msg="Mb Loss Before selection")
            self.mem_cnt = 0
            print("ring buffer is full, re-estimating of the constrains, we are at task", t)
            self.old_mem_grads = None

            self.select_samples_per_group(t)
