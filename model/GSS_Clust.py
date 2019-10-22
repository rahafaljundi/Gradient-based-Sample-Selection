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
import scipy as sp
import scipy.sparse as spa
from .common import MLP, ResNet18



# Auxiliary functions
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
        self.batch_size=args.batch_size
        self.n_iter = args.n_iter

        self.change_th = args.change_th  # gradient direction change threshold to re-select constraints
        # allocate ring buffer
        self.memory_data = torch.FloatTensor(self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(self.n_memories)
        self.added_index = self.n_sampled_memories
        # allocate  selected  memory
        self.sampled_memory_data = None
        self.sampled_memory_labs = None
        # allocate selected constraints
        self.constraints_data = None
        self.constraints_labs = None
        self.cluster_distance = 0
        self.age_weight = args.age
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
        # self.grads = torch.Tensor(sum(self.grad_dims), n_tasks*self.n_sampled_memories)
        # initialize a tensor for storing temprorarly the gradient of the current task
        # self.mem_grads = torch.Tensor(sum(self.grad_dims), 1)

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

    def get_samples_gradients(self,mem_data,mem_labs):

        mem_grads=None
        for x, y in zip(mem_data, mem_labs):
            self.zero_grad()
            ptloss = self.ce(self.forward(x.unsqueeze(0)), y.unsqueeze(0))
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            mem_grads = add_memory_grad(self.parameters, mem_grads, self.grad_dims)
        self.zero_grad()
        return mem_grads

    def select_k_centers(self,beta=0.1,alpha=1):
        self.eval()
        if self.sampled_memory_data is None:


            self.sampled_memory_data=self.memory_data[0].unsqueeze(0).clone()
            self.sampled_memory_labs = self.memory_labs[0].unsqueeze(0).clone()
            new_memories_data=self.memory_data[1:].clone()
            new_memories_labs = self.memory_labs[1:].clone()
        else:
            new_memories_data=self.memory_data.clone()
            new_memories_labs = self.memory_labs.clone()

        new_mem_features = self.get_samples_gradients(new_memories_data,new_memories_labs).clone()
        samples_mem_features = self.get_samples_gradients(self.sampled_memory_data,self.sampled_memory_labs).clone()
        new_dist=self.cosine_similarity(new_mem_features, samples_mem_features)
        #intra_distance
        if self.cluster_distance==0:

            intra_dist=self.cosine_similarity(samples_mem_features)
            max_dis=torch.max(intra_dist)

            eye=(torch.eye(intra_dist.size(0))*max_dis)
            self.cluster_distance=alpha*torch.min(intra_dist+eye)#

        added_indes=[]
        for new_mem_index in range(new_mem_features.size(0)):

            if torch.min(new_dist[new_mem_index])>self.cluster_distance:
                added_indes.append(new_mem_index)

        print("length of added inds",len(added_indes))
        if (len(added_indes)+self.sampled_memory_data.size(0))>self.n_sampled_memories:

            init_points=torch.cat((self.sampled_memory_data,new_memories_data[added_indes]),dim=0).clone()
            init_points_labels=torch.cat((self.sampled_memory_labs,new_memories_labs[added_indes]),dim=0).clone()
            init_points_feat=torch.cat((samples_mem_features,new_mem_features[added_indes]),dim=0).clone()
            est_mem_size=init_points_feat.size(0)
            init_feat_dist=self.cosine_similarity(init_points_feat)

            self.cluster_distance = torch.min(init_feat_dist+torch.eye(init_feat_dist.size(0))*torch.max(init_feat_dist))

            while est_mem_size>self.n_sampled_memories:
                self.cluster_distance=self.cluster_distance+beta*self.cluster_distance
                first_ind=torch.randint(0,init_points_feat.size(0),(1,))[0]
                #cent_feat=init_points_feat[first_ind].clone()
                cent_inds=[first_ind.item()]
                for feat_indx in range(init_points_feat.size(0)) :

                    if torch.min(init_feat_dist[feat_indx][cent_inds])>self.cluster_distance:
                        cent_inds.append(feat_indx)


                est_mem_size=len(cent_inds)
            print("BUFFER SIZE,",est_mem_size)
            self.sampled_memory_data=init_points[cent_inds].clone()
            self.sampled_memory_labs = init_points_labels[cent_inds].clone()
        else:
            self.sampled_memory_data=torch.cat((self.sampled_memory_data,new_memories_data[added_indes]),dim=0).clone()
            self.sampled_memory_labs=torch.cat((self.sampled_memory_labs,new_memories_labs[added_indes]),dim=0).clone()

        self.train()


    def cosine_similarity(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)

        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        sim= torch.mm(x1, x2.t())/(w1 * w2.t()) #, w1  # .clamp(min=eps), 1/cosinesim

        return 1-sim



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
            #self.opt.step()
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0
            print("ring buffer is full, re-estimating of the constrains, we are at task", t)
            self.old_mem_grads = None
            self.cosine_sim = [1] * self.n_constraints
            self.select_k_centers(self.age_weight)
