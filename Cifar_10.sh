#!/usr/bin/bash



CIFAR_10i="--n_layers 2 --n_hiddens 100 --data_path ./data/ --save_path results/cifar10 --batch_size 10 --log_every 10 --samples_per_task 10000 --data_file cifar10.pt   --tasks_to_preserve 5        --cuda yes "

mkdir results/cifar10
MY_PYTHON="python"
cd data/

cd raw/

$MY_PYTHON raw.py

cd ..

$MY_PYTHON cifar10.py \
	--o cifar10.pt \
	--seed 0 \
	--n_tasks 5

cd ..



echo "***********************GEM***********************"
$MY_PYTHON main.py $CIFAR_10i --model gem --lr 0.01 --n_memories 260 --memory_strength 0.5 --seed 0
$MY_PYTHON main.py $CIFAR_10i --model gem --lr 0.01 --n_memories 260 --memory_strength 0.5 --seed 1
$MY_PYTHON main.py $CIFAR_10i --model gem --lr 0.01 --n_memories 260 --memory_strength 0.5 --seed 2

echo "***********************Signle***********************"
$MY_PYTHON main.py $CIFAR_10i --model single --lr 0.01 --seed 0
$MY_PYTHON main.py $CIFAR_10i --model single --lr 0.01 --seed 1
$MY_PYTHON main.py $CIFAR_10i --model single --lr 0.01 --seed 2

echo "***********************GSS_Clust***********************"

$MY_PYTHON main.py $CIFAR_10i --model GSS_Clust --lr 0.01  --n_memories 100 --n_sampled_memories 1000  --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 0 --subselect 1 --age 0.01
$MY_PYTHON main.py $CIFAR_10i --model GSS_Clust --lr 0.01  --n_memories 100 --n_sampled_memories 1000  --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 1 --subselect 1 --age 0.01
$MY_PYTHON main.py $CIFAR_10i --model GSS_Clust --lr 0.01  --n_memories 100 --n_sampled_memories 1000  --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 2 --subselect 1 --age 0.01

echo "***********************FSS_Clust***********************"
$MY_PYTHON main.py $CIFAR_10i --model FSS_Clust --lr 0.01  --n_memories 100 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 0 --subselect 0
$MY_PYTHON main.py $CIFAR_10i --model FSS_Clust --lr 0.01  --n_memories 100 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 1 --subselect 0
$MY_PYTHON main.py $CIFAR_10i --model FSS_Clust --lr 0.01  --n_memories 100 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 2 --subselect 0

echo "***********************Rand***********************"
$MY_PYTHON main.py $CIFAR_10i --model rehearse_per_batch_rand --lr 0.01  --n_memories 100 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 0 --subselect 0
$MY_PYTHON main.py $CIFAR_10i --model rehearse_per_batch_rand --lr 0.01  --n_memories 100 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 1 --subselect 0
$MY_PYTHON main.py $CIFAR_10i --model rehearse_per_batch_rand --lr 0.01  --n_memories 100 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed 2 --subselect 0

echo "***********************GSS_Greedy***********************"
$MY_PYTHON main.py $CIFAR_10i --model Gradient_rehearsal++ --lr 0.01  --n_memories 10 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 10  --n_iter 10   --change_th 0. --seed 0  --subselect 1
$MY_PYTHON main.py $CIFAR_10i --model Gradient_rehearsal++ --lr 0.01  --n_memories 10 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 10  --n_iter 10   --change_th 0. --seed 1  --subselect 1
$MY_PYTHON main.py $CIFAR_10i --model Gradient_rehearsal++ --lr 0.01  --n_memories 10 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 10  --n_iter 10   --change_th 0. --seed 2  --subselect 1



