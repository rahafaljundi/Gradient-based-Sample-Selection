#!/usr/bin/bash
results="./results/permuted_Mnist/"
MNIST_PERM="--n_layers 2 --n_hiddens 100 --data_path ./data/ --save_path $results --batch_size 10 --log_every 10 --samples_per_task 1000 --data_file mnist_permutations.pt --cuda no  --tasks_to_preserve 10"

mkdir $results

MY_PYTHON="python"
cd data/

cd raw/

$MY_PYTHON raw.py

cd ..

$MY_PYTHON mnist_permutations.py \
	--o mnist_permutations.pt \
	--seed 0 \
	--n_tasks 10



cd ..

nb_seeds=2
seed=0
while [ $seed -le $nb_seeds ]
do

	echo $seed

	echo "***********************GEM***********************"
	$MY_PYTHON main.py $MNIST_PERM --model gem --lr 0.05 --n_memories 60 --memory_strength 0.5 --seed $seed


	echo "***********************Signle***********************"
	$MY_PYTHON main.py $MNIST_PERM --model single --lr 0.05 --seed $seed

	echo "***********************GSS_Clust***********************"

	$MY_PYTHON main.py $MNIST_PERM --model GSS_Clust --lr 0.05  --n_memories 100 --n_sampled_memories 300  --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 1 --age 0.01

	echo "***********************FSS_Clust***********************"
	$MY_PYTHON main.py $MNIST_PERM --model FSS_Clust --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 0

	echo "***********************Rand***********************"
	$MY_PYTHON main.py $MNIST_PERM --model rehearse_per_batch_rand --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 0

	echo "***********************GSS_Greedy***********************"
	$MY_PYTHON main.py $MNIST_PERM --model GSS_Greedy --lr 0.05  --n_memories 10 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 1 --age 0 --memory_strength 10

	echo "***********************GSS_IQP***********************"
	$MY_PYTHON main.py $MNIST_PERM --model GSS_IQP_Rehearse --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 0 --age 0 --memory_strength 0 --change_th 0.0

	((seed++))
done

$MY_PYTHON stats_calculater.py  $results 10
