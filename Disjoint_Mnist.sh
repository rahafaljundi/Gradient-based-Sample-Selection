#!/usr/bin/bash
results="./results/Disjoint_Mnist_5/"
MNIST_Split="--n_layers 2 --n_hiddens 100 --data_path ./data/ --save_path $results --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_split.pt --cuda no  --tasks_to_preserve 4"
MY_PYTHON="python"
mkdir $results
cd data/

cd raw/

$MY_PYTHON raw.py

cd ..



$MY_PYTHON mnist_split.py \
	--o mnist_split.pt \
	--seed 0 \
	--n_tasks 5

cd ..



nb_seeds=2
seed=0
while [ $seed -le $nb_seeds ]
do
	echo $seed

	echo "***********************iCaRL***********************"
	$MY_PYTHON main.py $MNIST_Split --model icarl --lr 0.1 --n_memories 300  --n_iter 3 --memory_strength 1 --seed $seed

	echo "***********************GEM***********************"
	$MY_PYTHON main.py $MNIST_Split --model gem --lr 0.05 --n_memories 60 --memory_strength 0.5 --seed $seed

	echo "***********************Single***********************"
	$MY_PYTHON main.py $MNIST_Split --model single --lr 0.05 --seed $seed

	echo "***********************GSS_Clust***********************"

	$MY_PYTHON main.py $MNIST_Split --model GSS_Clust --lr 0.05  --n_memories 100 --n_sampled_memories 300  --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 1 --age 0.01

	echo "***********************FSS_Clust***********************"
	$MY_PYTHON main.py $MNIST_Split --model FSS_Clust --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 0


	echo "***********************Rand***********************"
	$MY_PYTHON main.py $MNIST_Split --model rehearse_per_batch_rand --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 0

	echo "***********************GSS_Greedy***********************"
	$MY_PYTHON main.py $MNIST_Split --model GSS_Greedy --lr 0.05  --n_memories 10 --n_sampled_memories 300 --n_constraints 10 --memory_strength 10  --n_iter 3  --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 1 --age 0 --memory_strength 10 

	echo "***********************GSS_IQP***********************"
	$MY_PYTHON main.py $MNIST_Split --model GSS_IQP_Rehearse --lr 0.05  --n_memories 100 --n_sampled_memories 300 --n_constraints 10 --memory_strength 0  --n_iter 3 --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize  no --seed $seed --subselect 0 --age 0 --memory_strength 0 --change_th 0.0

	((seed++))
done

$MY_PYTHON stats_calculater.py  $results 5





