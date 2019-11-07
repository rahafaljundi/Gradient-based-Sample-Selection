## Gradient-based-Sample-Selection

Code for paper:
Gradient based sample selection for online continual learning
Rahaf Aljundi, Min Lin, Baptiste Goujaud, Yoshua Bengio. 
Neurips 2019

## (key) Requirements 

- Python 2.8 or more.
- Pytorch 1.1.0

`pip install -r requirements.txt`
or
`conda install --file requirements.txt`

for GSS_IQP, please install MIXEDIP package: https://github.com/oxfordcontrol/miosqp following their README file.
Note that for this you need to install quadprog and osqp

---- conda install -c omnia quadprog

---- conda install -c conda-forge/label/gcc7 osqp  %% if it doesn't work check this https://anaconda.org/conda-forge/osqp

Demo:

For disjoint MNIST please run ./Disjoint_Mnist.sh

For permuted MNIST please run ./Permuted_Mnist.sh

For disjoint CIFAR-10 please run ./Cifar_10.sh


