# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import datetime
import argparse
import random
import uuid
import time
import os
import pdb
import numpy as np

import torch
from metrics.metrics import confusion_matrix

# continuum iterator #########################################################


def load_datasets(args):

    print("path",args.data_path + '/' + args.data_file)
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max().item())
        n_outputs = max(n_outputs, d_te[i][2].max().item())
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


class Continuum:

    def __init__(self, data, args):
        self.data = data
        self.batch_size = args.batch_size
        n_tasks = len(data)

        task_permutation = range(n_tasks)

        if args.shuffle_tasks == 'yes':
            task_permutation = torch.randperm(n_tasks).tolist()

        sample_permutations = []

        samples_per_task= args.samples_per_task.split("-")
        n=1000
        for t in range(n_tasks):
            N=data[t][1].size(0)
            if len(samples_per_task)>t :

                if int(samples_per_task[t])<= 0:
                    n = N
                else:
                    n = min(int(samples_per_task[t]),N)
            else:
                n = min(n, N)
            print("*********Task",t,"Samples are",n)
            p = torch.randperm(data[t][1].size(0))[0:n]
            sample_permutations.append(p)

        self.permutation = []

        for t in range(n_tasks):
            task_t = task_permutation[t]

            for _ in range(args.n_epochs):
                task_p = [[task_t, i] for i in sample_permutations[task_t]]
                random.shuffle(task_p)
                self.permutation += task_p

        self.length = len(self.permutation)
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            ti = self.permutation[self.current][0]
            j = []
            i = 0
            while (((self.current + i) < self.length) and
                   (self.permutation[self.current + i][0] == ti) and
                   (i < self.batch_size)):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            return self.data[ti][1][j], ti, self.data[ti][2][j]

# train handle ###############################################################




def eval_tasks(model, tasks,current_task, args):
    """
    it also evaluates the performance of the model on samples from all the tasks and gives the average performance on all the samples regardless of their task
    :param model:
    :param tasks:
    :param args:
    :return:
    """
    model.eval()
    result = []
    total_size=0
    total_pred=0
    current_result = []
    current_avg_acc = 0
    for i, task in enumerate(tasks):

        t = i
        x = task[1]
        y = task[2]
        rt = 0

        eval_bs = x.size(0)

        for b_from in range(0, x.size(0), eval_bs):
            b_to = min(b_from + eval_bs, x.size(0) - 1)
            if b_from == b_to:
                xb = x[b_from].view(1, -1)
                yb = torch.LongTensor([y[b_to]]).view(1, -1)
            else:
                xb = x[b_from:b_to]
                yb = y[b_from:b_to]
            if args.cuda:
                xb = xb.cuda()
            _, pb = torch.max(model(xb, t).data.cpu(), 1, keepdim=False)
            rt += (pb == yb).float().sum()


        result.append(rt / x.size(0))

        total_size+= x.size(0)
        total_pred+=rt

        if t == current_task:
            current_result=[res for res in result]
            current_avg_acc=total_pred / total_size


    print("###################### EVAL BEGIN ##########################")
    print(result)

    print("###################### EVAL ENDS ##########################")
    torch.save(( model.state_dict(),current_result,current_avg_acc), model.fname + '.pt')

    return result,total_pred/total_size,current_avg_acc,current_avg_acc

def eval_on_memory(args):
    """
    compute accuracy on the buffer
    :return:
    """
    model.eval()
    acc_on_mem=0
    if 'yes' in args.eval_memory:
        for x,y in zip(model.sampled_memory_data,model.sampled_memory_labs):
            if args.cuda:
                x = x.cuda()
            _, pb = torch.max(model(x.unsqueeze(0)).data.cpu(), 1, keepdim=False)

            acc_on_mem += (pb == y.data.cpu()).float()

        acc_on_mem=(acc_on_mem/model.sampled_memory_data.size(0))
    return acc_on_mem

def life_experience(model, continuum, x_te, args):
    result_a = []
    result_t = []
    result_all=[]#avg performance on all test samples
    current_res_per_t=[]#per task accuracy up until the current task
    current_avg_acc_list=[]#avg accuracy on task seen so far
    current_task = 0
    time_start = time.time()

    for (i, (x, t, y)) in enumerate(continuum):
        if t>args.tasks_to_preserve:
            break
        if(((i % args.log_every) == 0) or (t != current_task)):

            res_per_t,res_all,current_result,current_avg_acc=eval_tasks(model, x_te,current_task, args)
            result_a.append(res_per_t)
            result_all.append(res_all)
            result_t.append(current_task)
            current_res_per_t.append(current_result)
            current_avg_acc_list.append(current_avg_acc)
            current_task = t

        v_x = x.view(x.size(0), -1)
        v_y = y.long()

        if args.cuda:
            v_x = v_x.cuda()
            v_y = v_y.cuda()

        model.train()
        model.observe(v_x, t, v_y)

    res_per_t, res_all,current_result,current_avg_acc = eval_tasks(model, x_te, args.tasks_to_preserve,args)
    res_on_mem=eval_on_memory(args)
    result_a.append(res_per_t)
    result_t.append(current_task)
    current_res_per_t.append(current_result)#at the end those are similar to the previous two
    current_avg_acc_list.append(current_avg_acc)
    result_all.append(res_all)

    time_end = time.time()
    time_spent = time_end - time_start

    return torch.Tensor(result_t), torch.Tensor(result_a),torch.Tensor(result_all),torch.Tensor(current_res_per_t),torch.Tensor(current_avg_acc_list),res_on_mem, time_spent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuum learning')

    # model parameters
    parser.add_argument('--model', type=str, default='single',
                        help='model to train')
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--shared_head', type=str, default='yes',
                        help='shared head between tasks')
    parser.add_argument('--bias', type=int, default='1',
                        help='do we add bias to the last layer? does that cause problem?')
    # memory parameters
    parser.add_argument('--n_memories', type=int, default=0,
                        help='number of memories per task')
    parser.add_argument('--n_sampled_memories', type=int, default=0,
                        help='number of sampled_memories per task')
    parser.add_argument('--n_constraints', type=int, default=0,
                        help='number of constraints to use during online training')
    parser.add_argument('--b_rehearse', type=int, default=0,
                        help='if 1 use mini batch while rehearsing')
    parser.add_argument('--tasks_to_preserve', type=int, default=1,
                        help='number of tasks to preserve')
    parser.add_argument('--change_th', type=float, default=0.0,
                        help='gradients similarity change threshold for re-estimating the constraints')
    parser.add_argument('--slack', type=float, default=0.01,
                        help='slack for small gradient norm')
    parser.add_argument('--normalize', type=str, default='no',
                        help='normalize gradients before selection')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--finetune', default='no', type=str,
                        help='whether to initialize nets in indep. nets')

    # optimizer parameters
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--n_iter', type=int, default=1,
                        help='Number of iterations per batch')
    parser.add_argument('--repass', type=int, default=0,
                        help='make a repass over the previous da<ta')

    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--mini_batch_size', type=int, default=0,
                        help='mini batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='SGD learning rate')

    # experiment parameters
    parser.add_argument('--cuda', type=str, default='no',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')
    parser.add_argument('--output_name', type=str, default='',
                        help='special output name for the results?')
    # data parameters
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--data_file', default='mnist_permutations.pt',
                        help='data file')
    parser.add_argument('--samples_per_task', type=str, default='-1',
                        help='training samples per task (all if negative)')

    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')
    parser.add_argument('--eval_memory', type=str, default='no',
                        help='compute accuracy on memory')


    parser.add_argument('--age', type=float, default=1,
                        help='consider age for sample selection')

    parser.add_argument('--subselect', type=int, default=1,
                        help='first subsample from recent memories')
    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.finetune = True if args.finetune == 'yes' else False
    args.normalize = True if args.normalize == 'yes' else False
    args.shared_head = True if args.shared_head == 'yes' else False
    if args.mini_batch_size==0:
        args.mini_batch_size=args.batch_size#no mini iterations
    # multimodal model has one extra layer
    if args.model == 'multimodal':
        args.n_layers -= 1

    # unique identifier
    uid = uuid.uuid4().hex

    # initialize seeds
    print("seed is", args.seed)
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    # load data

    x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)

    # set up continuum
    continuum = Continuum(x_tr, args)
    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)

    #set up file name for inbetween saving
    if args.n_sampled_memories==0:
        args.n_sampled_memories=args.n_memories
    if args.output_name:
        model.fname=args.output_name
    else:
        model.fname = args.model+'D'+str(args.samples_per_task)+'Mb'+str(args.n_sampled_memories) + '_Ma' +str(args.n_constraints) +'_St'+str(args.memory_strength)+ '_Ch'+str(args.change_th) + '_nit' +str(args.n_iter) +'_slack'+str( args.slack) +'_normalize'+str( args.normalize)+ '_'+ args.data_file + '_'
    model.fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model.fname += '_' + uid
    model.fname = os.path.join(args.save_path,model.fname)

    if args.cuda:
        model.cuda()
    if args.shared_head:
        model.is_cifar=False
        model.nc_per_task = n_outputs
    # run model on continuum
    result_t, result_a, avg_accuracy,current_res_per_t,current_avg,accurcy_on_mem ,spent_time= life_experience(
        model, continuum, x_te, args)

    # prepare saving path and file name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


    # save confusion matrix and print one line of stats
    stats = confusion_matrix(result_t, result_a,avg_accuracy,accurcy_on_mem,args.tasks_to_preserve, model.fname+'.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.3f" % stat for stat in stats])
    print(model.fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    torch.save((result_t, result_a, model.state_dict(),current_res_per_t,current_avg,
                stats, one_liner, args), model.fname + '.pt')
