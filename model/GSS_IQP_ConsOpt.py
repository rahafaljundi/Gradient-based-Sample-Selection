# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
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


def cosine_similarity_selector_IQP_Exact(x1, solver, nb_selected, eps=1e-3, slack=0.01):
    # your code

    """
    Integer programming
    """

    # x1=gradient memories

    start_time = time.time()
    x2 = None

    w1 = x1.norm(p=2, dim=1, keepdim=True)

    inds = torch.nonzero(torch.gt(w1, slack))[:, 0]
    print("removed due to gradients", w1.size(0) - inds.size(0))
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
    try:
        solver.setup(G, a, C, h_final_lower, h_final_upper, idx, hlower, hupper, miosqp_settings, osqp_settings)
    except:
        import pdb
        pdb.set_trace()
    results = solver.solve()
    elapsed_time = time.time() - start_time
    print("ELAPSED TIME IS ", elapsed_time)
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

def cosine_similarity(self, x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)

    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()), w1  # .clamp(min=eps)
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

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.n_sampled_memories = args.n_sampled_memories
        self.n_constraints = args.n_constraints
        self.gpu = args.cuda
        self.repass = args.repass
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.slack = args.slack
        self.change_th = args.change_th  # gradient direction change threshold to re-select constraints
        # allocate ring buffer
        self.memory_data = torch.FloatTensor(self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(self.n_memories)
        # allocate  selected  memory
        self.sampled_memory_data = None
        self.sampled_memory_labs = None
        # allocate selected constraints
        self.constraints_data = None
        self.constraints_labs = None
        self.subselect = args.subselect  # first select from recent memory and then add to samples memories
        # old grads to measure changes
        self.old_mem_grads = None
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

    def select_samples_per_group(self):
        """
        Assuming a ring buffer, backup constraints and constrains,
        re-estimate the backup constrains and constrains

        """

        print("constraints selector")
        self.mem_grads = None
        # get gradients from the ring buffer
        for x, y in zip(self.memory_data, self.memory_labs):
            self.zero_grad()
            ptloss = self.ce(self.forward(x.unsqueeze(0)), y.unsqueeze(0))
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            self.mem_grads = add_memory_grad(self.parameters, self.mem_grads, self.grad_dims)

        added_inds = torch.arange(self.n_memories)
        from_buffer_size = added_inds.size(0)
        self.new_mem_grads = self.mem_grads[added_inds].clone()
        # estimate the active constraints from the backup samples
        self.mem_grads = None
        if not self.sampled_memory_data is None:
            for x, y in zip(self.sampled_memory_data, self.sampled_memory_labs):
                self.zero_grad()
                ptloss = self.ce(self.forward(x.unsqueeze(0)), y.unsqueeze(0))
                ptloss.backward()
                # add the new grad to the memory grads and add it is cosine similarity
                self.mem_grads = add_memory_grad(self.parameters, self.mem_grads, self.grad_dims)
            # select main constrains

            inds = cosine_similarity_selector_IQP_Exact(self.mem_grads, nb_selected=self.n_constraints,
                                                        solver=self.solver, slack=self.slack)

            self.constraints_data = self.sampled_memory_data[inds].clone()
            self.constraints_labs = self.sampled_memory_labs[inds].clone()

            print("selected labels are", self.constraints_labs)
            # self.compute_similarity_between_classes(self.mem_grads,self.sampled_memory_labs)

            self.sampled_memory_data = torch.cat((self.memory_data[added_inds], self.sampled_memory_data),
                                                 dim=0).clone()
            self.sampled_memory_labs = torch.cat((self.memory_labs[added_inds], self.sampled_memory_labs,),
                                                 dim=0).clone()

            self.mem_grads = torch.cat((self.new_mem_grads, self.mem_grads), dim=0)

            if self.n_sampled_memories < self.mem_grads.size(0):

                inds = cosine_similarity_selector_IQP_Exact(self.mem_grads, nb_selected=self.n_sampled_memories,
                                                            solver=self.solver, slack=self.slack)
                print("number of retained memories", torch.nonzero(torch.ge(inds, from_buffer_size)).size(0))
                if torch.nonzero(torch.ge(inds, from_buffer_size)).size(0) == 0:
                    pdb.set_trace()
                self.sampled_memory_data = self.sampled_memory_data[inds].clone()
                self.sampled_memory_labs = self.sampled_memory_labs[inds].clone()
                print("selected Buffer labels are", self.sampled_memory_labs)


        else:
            self.sampled_memory_data = self.memory_data[added_inds].clone()
            self.sampled_memory_labs = self.memory_labs[added_inds].clone()
        labels = torch.unique(self.sampled_memory_labs)
        for lab in labels:
            print("number of samples from class", lab, torch.nonzero(torch.eq(self.sampled_memory_labs, lab)).size(0))
        self.mem_grads = None




    def select_samples_from_backup(self):
        """
        Assuming a buffer of stored samples
        re-estimate the constraints

        """

        print("active constraints selector")

        self.mem_grads = None

        for x, y in zip(self.sampled_memory_data, self.sampled_memory_labs):
            self.zero_grad()
            ptloss = self.ce(self.forward(x.unsqueeze(0)), y.unsqueeze(0))
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            self.mem_grads = add_memory_grad(self.parameters, self.mem_grads, self.grad_dims)

        # select main constrains
        back_up_inds = cosine_similarity_selector_IQP_Exact(self.mem_grads, nb_selected=self.n_constraints,
                                                            solver=self.solver, slack=self.slack)
        self.constraints_data = self.sampled_memory_data[back_up_inds].clone()
        self.constraints_labs = self.sampled_memory_labs[back_up_inds].clone()
        print("selected labels are", self.constraints_labs)
        self.mem_grads = None
        self.old_mem_grads = None





    def forward_constraints(self):
        self.mem_grads = None
        if self.constraints_data is not None:

            for i in range(self.constraints_data.size(0)):


                self.zero_grad()
                ptloss = self.ce(self.forward(self.constraints_data[i].unsqueeze(0)),
                                 self.constraints_labs[i].unsqueeze(0))
                ptloss.backward()
                this_grad_vec = get_grad_vector(self.parameters, self.grad_dims)
                if self.mem_grads is None:
                    self.mem_grads = this_grad_vec.unsqueeze(1).clone()
                else:
                    self.mem_grads = torch.cat((self.mem_grads, this_grad_vec.unsqueeze(1)), dim=1).clone()

            if self.mem_grads is not None and self.old_mem_grads is None:
                print("Updating old grds, similarity must be one!")

                self.old_mem_grads = self.mem_grads.clone()




    # MAIN TRAINING FUNCTION
    def observe(self, x, t, y):
        # update memory
        # temp
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

        for iter_i in range(self.n_iter):

            # get gradients on previous constraints
            self.forward_constraints()

            # check the chnages in the gradients of the samples
            if self.old_mem_grads is not None:

                self.cosine_sim = [None] * self.old_mem_grads.size(1)
                for index in range(self.mem_grads.size(1)):
                    cosine_sim, norm = cosine_similarity(self.mem_grads[:, index].unsqueeze(0),
                                                              self.old_mem_grads[:, index].unsqueeze(0))
                    self.cosine_sim[index] = cosine_sim.item() if norm > self.slack else 1
                str_osine_sim = ['%.2f' % elem for elem in self.cosine_sim]
                print("minimum gradients_cosine_similarity", min(self.cosine_sim))
                if min(self.cosine_sim) < self.change_th:
                    _, index = torch.min(torch.tensor(self.cosine_sim), dim=0)

                    print(
                        "Restimating the Active Constraints to preserve")  # ,torch.norm(self.old_mem_grads[:,index],p=2)

                    self.old_mem_grads = None
                    self.select_samples_from_backup()
                    self.forward_constraints()
            # now compute the grad on the current minibatch
            self.zero_grad()

            loss = self.ce(self.forward(x), y)
            loss.backward()

            # check if gradient violates constraints
            if self.constraints_data is not None:
                # copy gradient

                this_grad_vec = get_grad_vector(self.parameters, self.grad_dims)

                dotp = torch.mm(this_grad_vec.unsqueeze(0),
                                self.mem_grads)
                if (dotp < 0).sum() != 0:
                    # print("number of violated samples",(dotp < 0).sum())
                    this_grad_vec = this_grad_vec.unsqueeze(1)

                    project2cone2(this_grad_vec, self.mem_grads, self.margin)

                    # copy gradients back
                    overwrite_grad(self.parameters, this_grad_vec,
                                   self.grad_dims)
            self.opt.step()

        if self.mem_cnt == self.n_memories:
            if self.repass > 0 and self.sampled_memory_data is not None:
                print("repassing")
                self.zero_grad()
                self.rehearse_st(self.sampled_memory_data, self.sampled_memory_labs)
            self.mem_cnt = 0
            print("ring buffer is full, re-estimating of the constrains, we are at task", t)
            self.old_mem_grads = None
            self.cosine_sim = [1] * self.n_constraints
            self.select_samples_per_group()
