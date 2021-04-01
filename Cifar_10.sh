#!/usr/bin/bash


results="./results/cifar10"
CIFAR_10i="--n_layers 2 --n_hiddens 100 --data_path ./data/ --save_path $results --batch_size 10 --log_every 10 --samples_per_task 10000 --data_file cifar10.pt   --tasks_to_preserve 5        --cuda yes "

mkdir $results
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

nb_seeds=2
seed=0
while [ $seed -le $nb_seeds ]
do

	echo $seed

	echo "***********************iCaRL***********************"
	$MY_PYTHON main.py $CIFAR_10i --model icarl --lr 0.1 --n_memories 1000  --n_iter 3 --memory_strength 1 --seed $seed

	echo "***********************GEM***********************"
	$MY_PYTHON main.py $CIFAR_10i --model gem --lr 0.01 --n_memories 260 --memory_strength 0.5 --seed $seed


	echo "***********************Signle***********************"
	$MY_PYTHON main.py $CIFAR_10i --model single --lr 0.01 --seed $seed


	echo "***********************GSS_Clust***********************"

	#$MY_PYTHON main.py $CIFAR_10i --model GSS_Clust --lr 0.01  --n_memories 100 --n_sampled_memories 1000  --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 1 --age 0.01


	echo "***********************FSS_Clust***********************"
	$MY_PYTHON main.py $CIFAR_10i --model FSS_Clust --lr 0.01  --n_memories 100 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 0


	echo "***********************Rand***********************"
	$MY_PYTHON main.py $CIFAR_10i --model rehearse_per_batch_rand --lr 0.01  --n_memories 100 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 0


	echo "***********************GSS_Greedy***********************"
	$MY_PYTHON main.py $CIFAR_10i --model GSS_Greedy  --lr 0.01  --n_memories 10 --n_sampled_memories 1000 --n_constraints 10 --memory_strength 10  --n_iter 10   --change_th 0. --seed $seed  --subselect 1

	((seed++))
done

$MY_PYTHON stats_calculater.py  $results 5



