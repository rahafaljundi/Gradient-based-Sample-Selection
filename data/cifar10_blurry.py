
import argparse
import os.path
import torch
import pdb
parser = argparse.ArgumentParser()

parser.add_argument('--i', default='raw/cifar10.pt', help='input directory')
parser.add_argument('--o', default='cifar10.pt', help='output file')
parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--p', default=0, type=float, help='percentage of samples of a given task in its turn ')
args = parser.parse_args()
assert(args.p<=1 and args.p>0)
torch.manual_seed(args.seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr, x_te, y_te = torch.load(os.path.join(args.i))
x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
x_te = x_te.float().view(x_te.size(0), -1) / 255.0

cpt = int(10 / args.n_tasks)
tasks_tr_inds={}
for t in range(args.n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt

    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
    tasks_tr_inds[t]=i_tr
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
    tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])
#assign inds to the tasks
within_task_inds={}
blurred_inds=[]

for t in tasks_tr_inds.keys():

    in_task_inds=torch.randperm(tasks_tr_inds[t].size(0))[0: int(tasks_tr_inds[t].size(0)*args.p)]
    within_task_inds[t]=tasks_tr_inds[t][in_task_inds]
    in_set_inds=set(within_task_inds[t].numpy().flatten())
    all_set_inds=set(tasks_tr_inds[t].numpy().flatten())
    out_set_inds=all_set_inds-in_set_inds
    out_task_inds=list(out_set_inds)
    blurred_inds.extend(out_task_inds)
    print("blurred inds len",len(blurred_inds))

blurred_inds=torch.tensor( blurred_inds)
blurred_inds=blurred_inds[torch.randperm(blurred_inds.size(0))]#shuffel the samples
per_task_blurred=int(blurred_inds.size(0)/args.n_tasks)
for t in range(args.n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt

    task_tr_x= x_tr[within_task_inds[t]].clone()
    task_tr_y = y_tr[within_task_inds[t]].clone()
    start=(t*per_task_blurred)
    end=start+per_task_blurred
    print("start",start,"end",end)
    this_out_inds=blurred_inds[torch.arange(start,end,dtype=torch.long)]
    task_tr_x=torch.cat((task_tr_x,x_tr[this_out_inds]))
    task_tr_y = torch.cat((task_tr_y, y_tr[this_out_inds]))
    final_shuffel=torch.randperm(task_tr_x.size(0))
    task_tr_x=task_tr_x[final_shuffel].clone()
    task_tr_y = task_tr_y[final_shuffel].clone()
    tasks_tr.append([(c1, c2), task_tr_x.clone(), task_tr_y.clone()])

torch.save([tasks_tr, tasks_te], args.o)
