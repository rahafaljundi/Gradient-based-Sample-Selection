# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

import numpy as np
import random
from .common import MLP, ResNet18


class Net(torch.nn.Module):
    # Re-implementation of
    # S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert.
    # iCaRL: Incremental classifier and representation learning.
    # CVPR, 2017.
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.nt = n_tasks
        self.reg = args.memory_strength
        self.n_memories = args.n_memories
        self.num_exemplars = 0
        self.n_feat = n_outputs
        self.n_classes = n_outputs
        self.samples_per_task = args.samples_per_task
        self.n_iter = args.n_iter
        self.samples_per_task=(int)(self.samples_per_task)
        if self.samples_per_task <= 0:
            error('set explicitly args.samples_per_task')
        self.examples_seen = 0

        # setup network
        nl, nh = args.n_layers, args.n_hiddens
        self.is_cifar = ('cifar10' in args.data_file)
        if self.is_cifar:
            self.net = ResNet18(n_outputs, bias=args.bias)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])


        # setup optimizer
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()#BCELoss
        self.kl = torch.nn.KLDivLoss()  # for distillation
        self.lsm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)

        # memory
        self.memx = None  # stores raw inputs, PxD
        self.memy = None
        self.mem_class_x = {}  # stores exemplars class by class
        self.mem_class_y = {}

        self.gpu = args.cuda
        self.nc_per_batch = int(n_outputs / n_tasks)
        self.n_outputs = n_outputs

    def compute_offsets(self, task):

        offset1 = task * self.nc_per_batch
        offset2 = (task + 1) * self.nc_per_batch
        return int(offset1), int(offset2)

    def forward(self, x, t):
        # nearest neighbor
        nd = self.n_feat
        ns = x.size(0)

        if t * self.nc_per_batch not in self.mem_class_x.keys():
            # no exemplar in memory yet, output uniform distr. over classes in
            # task t above, we check presence of first class for this task, we
            # should check them all
            out = torch.Tensor(ns, self.n_classes).fill_(-10e10)
            out[:, 0:self.n_classes].fill_(
                1.0 / self.n_classes)
            if self.gpu:
                out = out.cuda()
            return out
        means = torch.ones(len(self.mem_class_x.keys()), nd) * float('inf')
        if self.gpu:
            means = means.cuda()

        #for cc in range(offset1, offset2):
        #    means[cc -
        #          offset1] = self.net(self.mem_class_x[cc]).data.mean(0)

        for cc in self.mem_class_x.keys():
            means[cc] = self.net(self.mem_class_x[cc]).data.mean(0)
        classpred = torch.LongTensor(ns)
        preds = self.net(x).data.clone()
        for ss in range(ns):
            dist = (means - preds[ss].expand(len(self.mem_class_x.keys()), nd)).norm(2, 1)
            _, ii = dist.min(0)
            ii = ii.squeeze()
            classpred[ss] = ii.item()

        out = torch.zeros(ns, self.n_classes)
        if self.gpu:
            out = out.cuda()
        for ss in range(ns):
            out[ss, classpred[ss]] = 1
        return out  # return 1-of-C code, ns x nc

    def forward_training(self, x, t):

        output = self.net(x)
        # make sure we predict classes within the current task
        return output

    def observe(self, x, t, y):

        self.net.train()
        self.examples_seen += x.size(0)

        if self.memx is None:
            self.memx = x.data.clone()
            self.memy = y.data.clone()
        else:
            self.memx = torch.cat((self.memx, x.data.clone()))
            self.memy = torch.cat((self.memy, y.data.clone()))
        for iter in range(self.n_iter):
            self.net.zero_grad()
            loss = self.bce((self.net(x)),
                            y)

            if self.num_exemplars > 0:


                # distillation

                # first generate a minibatch with one example per class from
                # previous tasks

                inp_dist = torch.zeros(len(self.mem_class_x.keys()), x.size(1))
                target_dist = torch.zeros(len(self.mem_class_x.keys()), self.n_feat)

                if self.gpu:
                    inp_dist = inp_dist.cuda()
                    target_dist = target_dist.cuda()
                for cc in self.mem_class_x.keys():
                    indx = random.randint(0, len(self.mem_class_x[cc ]) - 1)
                    inp_dist[cc] = self.mem_class_x[cc ][indx].clone()
                    target_dist[cc] = self.mem_class_y[cc ][indx].clone()
                # Add distillation loss
                loss += self.reg * self.kl(
                    self.lsm(self.net(inp_dist)
                             ),
                    self.sm(target_dist))
            # bprop and update
            loss.backward()
            self.opt.step()

        # check whether this is the last minibatch of the current task
        # We assume only 1 epoch!

        if self.examples_seen == self.samples_per_task:

            self.examples_seen = 0
            # get labels from previous task; we assume labels are consecutive
            if self.gpu:
                all_labs = torch.LongTensor(np.unique(self.memy.cpu().numpy()))
            else:
                all_labs = torch.LongTensor(np.unique(self.memy.numpy()))
            num_classes = all_labs.size(0)
            #assert(num_classes == self.nc_per_task)
            # Reduce exemplar set by updating value of num. exemplars per class
            self.num_exemplars = int(self.n_memories /
                                     (num_classes + len(self.mem_class_x.keys())))

            for ll in range(num_classes):
                lab = all_labs[ll]
                if self.gpu:
                    lab = lab.cuda()

                indxs = (self.memy == lab).nonzero().squeeze()
                cdata = self.memx.index_select(0, indxs)

                # Construct exemplar set for last task
                mean_feature = self.net(cdata).data.clone().mean(0)
                nd = self.n_feat

                exemplars = torch.zeros(self.num_exemplars, x.size(1))
                if self.gpu:
                    exemplars = exemplars.cuda()
                ntr = cdata.size(0)
                # used to keep track of which examples we have already used
                taken = torch.zeros(ntr)
                model_output = self.net(cdata).data.clone()
                for ee in range(self.num_exemplars):
                    prev = torch.zeros(1, nd)
                    if self.gpu:
                        prev = prev.cuda()
                    if ee > 0:
                        prev = self.net(exemplars[:ee]).data.clone().sum(0)
                    cost = (mean_feature.expand(ntr, nd) - (model_output
                                                            + prev.expand(ntr, nd)) / (ee + 1)).norm(2, 1).squeeze()
                    _, indx = cost.sort(0)
                    winner = 0
                    while winner < indx.size(0) and taken[indx[winner]] == 1:
                        winner += 1
                    if winner < indx.size(0):
                        taken[indx[winner]] = 1
                        exemplars[ee] = cdata[indx[winner]].clone()
                    else:
                        exemplars = exemplars[:indx.size(0), :].clone()
                        self.num_exemplars = indx.size(0)
                        break
                # update memory with exemplars
                self.mem_class_x[lab.item()] = exemplars.clone()

            # recompute outputs for distillation purposes

            for cc in self.mem_class_x.keys():
                #reduce number of examplers

                self.mem_class_x[cc]=self.mem_class_x[cc][0:self.num_exemplars]
                self.mem_class_y[cc] = self.net(
                    self.mem_class_x[cc]).data.clone()
            self.memx = None
            self.memy = None
