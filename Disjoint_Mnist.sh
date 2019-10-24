#!/usr/bin/bash

MNIST_Split="--n_layers 2 --n_hiddens 100 --data_path ./data/ --save_path results/Disjoint_Mnist_5/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_split.pt --cuda no  --tasks_to_preserve 4"
MY_PYTHON="python"

echo "iCaRL"
$MY_PYTHON main.py $MNIST_Split --model icarl --lr 0.1 --n_memories 300  --n_iter 3 --memory_strength 1 --seed 0
$MY_PYTHON main.py $MNIST_Split --model icarl --lr 0.1 --n_memories 300  --n_iter 3 --memory_strength 1 --seed 1
$MY_PYTHON main.py $MNIST_Split --model icarl --lr 0.1 --n_memories 300  --n_iter 3 --memory_strength 1 --seed 2

echo "GEM"
$MY_PYTHON main.py $MNIST_Split --model gem --lr 0.05 --n_memories 60 --memory_strength 0.5 --seed 0
$MY_PYTHON main.py $MNIST_Split --model gem --lr 0.05 --n_memories 60 --memory_strength 0.5 --seed 1
$MY_PYTHON main.py $MNIST_Split --model gem --lr 0.05 --n_memories 60 --memory_strength 0.5 --seed 2

echo "Signle"
$MY_PYTHON main.py $MNIST_Split --model single --lr 0.05 --seed 0
$MY_PYTHON main.py $MNIST_Split --model single --lr 0.05 --seed 1
$MY_PYTHON main.py $MNIST_Split --model single --lr 0.05 --seed 2

echo "GSS_Clust"

$MY_PYTHON main.py $MNIST_Split --model GSS_Clust --lr 0.05  --n_memories 100 --n_sampled_memories 300  --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 0 --subselect 1 --age 0.01
$MY_PYTHON main.py $MNIST_Split --model GSS_Clust --lr 0.05  --n_memories 100 --n_sampled_memories 300  --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 1 --subselect 1 --age 0.01
$MY_PYTHON main.py $MNIST_Split --model GSS_Clust --lr 0.05  --n_memories 100 --n_sampled_memories 300  --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 2 --subselect 1 --age 0.01

echo "FSS_Clust"
$MY_PYTHON main.py $MNIST_Split --model FSS_Clust --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 0 --subselect 0
$MY_PYTHON main.py $MNIST_Split --model FSS_Clust --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 1 --subselect 0
$MY_PYTHON main.py $MNIST_Split --model FSS_Clust --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 2 --subselect 0

echo "Rand"
$MY_PYTHON main.py $MNIST_Split --model rehearse_per_batch_rand --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 0 --subselect 0
$MY_PYTHON main.py $MNIST_Split --model rehearse_per_batch_rand --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 1 --subselect 0
$MY_PYTHON main.py $MNIST_Split --model rehearse_per_batch_rand --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 2 --subselect 0

echo "GSS_Greedy"
$MY_PYTHON main.py $MNIST_Split --model GSS_Greedy --lr 0.05  --n_memories 10 --n_sampled_memories 300 --n_constraints 10 --memory_strength 10  --n_iter 3  --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 0 --subselect 1 --age 0 --memory_strength 10 
$MY_PYTHON main.py $MNIST_Split --model GSS_Greedy --lr 0.05  --n_memories 10 --n_sampled_memories 300 --n_constraints 10 --memory_strength 10  --n_iter 3  --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 1 --subselect 1 --age 0 --memory_strength 10 
$MY_PYTHON main.py $MNIST_Split --model GSS_Greedy --lr 0.05  --n_memories 10 --n_sampled_memories 300 --n_constraints 10 --memory_strength 10  --n_iter 3  --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 2 --subselect 1 --age 0 --memory_strength 10 

echo "GSS_IQP"
$MY_PYTHON main.py $MNIST_Split --model GSS_IQP_Rehearse --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 0 --subselect 0 --age 0 --memory_strength 0 --change_th 0.0
$MY_PYTHON main.py $MNIST_Split --model GSS_IQP_Rehearse --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 0 --subselect 0 --age 0 --memory_strength 0 --change_th 0.0
$MY_PYTHON main.py $MNIST_Split --model GSS_IQP_Rehearse --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 0 --subselect 0 --age 0 --memory_strength 0 --change_th 0.0

