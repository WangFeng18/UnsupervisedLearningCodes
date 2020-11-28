### Implementation of Contrastive Learning Methods based on Memory Bank and BYOL
Run the Instance Discrimination (representative contrastive learning method) task by:
 
`python run.py --data './data/' --cudaenv '0,1,2,3' --gpus '0,1,2,3' --exp 'your_exp_path' --dataset cifar100 --batch_size 128 --lr 0.03 --network resnet18_cifar --t 0.30 --loss insd`

Run the BYOL by the following code:
 
`python byol.py --data './data/' --cudaenv '0,1,2,3' --gpus '0,1,2,3' --exp 'your_exp_path' --dataset cifar100 --batch_size 512 --lr 0.12 --network resnet18_cifar`

### CIFAR100 Results (with ResNet-18)
For Instance Discrimination task, the results on CIFAR100 with different temperatures (weight decay of 5e-4) are:

--------

Temporature  | 0.05 | 0.07 | 0.10|0.20|0.30|0.40|0.50|0.60|0.70|0.80|
--------- | --------| --- | --- | --- | --- | --- | --- | --- | --- | --- |
LinearAcc |49.80|51.82|52.46|56.05|**56.44**|55.47|54.17|53.05|50.99|50.08|

-------

For BYOL, I have only tested a simple set of hyper-parameters: using a fixed momentum of m=0.996 (instead of a cosine ramp up function in the original paper):

|BYOL|
|----|
|59.96|



