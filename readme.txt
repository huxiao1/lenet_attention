**Clean**:
lenet:
python model_train.py --dataset mnist --model lenet --batch_size 128 --epochs 300 --lr 0.0022 --patience 10 --num_classes 10 --save_dir /scratch/gilbreth/hu953/cs57800/pth
python model_train.py --dataset emnist --emnist_split balanced --model lenet --batch_size 128 --epochs 300 --lr 0.0022 --patience 10 --num_classes 47 --save_dir /scratch/gilbreth/hu953/cs57800/pth

lenet_with_transformer:
python model_train.py --dataset mnist --model transformer --batch_size 128 --epochs 300 --lr 0.0022 --patience 10 --num_classes 10 --save_dir /scratch/gilbreth/hu953/cs57800/pth
python model_train.py --dataset emnist --emnist_split balanced --model transformer --batch_size 128 --epochs 300 --lr 0.0022 --patience 10 --num_classes 47 --save_dir /scratch/gilbreth/hu953/cs57800/pth



**Blur**:
lenet:
python model_train.py --dataset mnist --model lenet --batch_size 128 --epochs 300 --lr 0.0022 --patience 10 --num_classes 10 --save_dir /scratch/gilbreth/hu953/cs57800/pth --noise_std 0.1
python model_train.py --dataset emnist --emnist_split balanced --model lenet --batch_size 128 --epochs 300 --lr 0.0022 --patience 10 --num_classes 47 --save_dir /scratch/gilbreth/hu953/cs57800/pth --noise_std 0.1

lenet_with_transformer:
python model_train.py --dataset mnist --model transformer --batch_size 128 --epochs 300 --lr 0.0022 --patience 10 --num_classes 10 --save_dir /scratch/gilbreth/hu953/cs57800/pth --noise_std 0.1
python model_train.py --dataset emnist --emnist_split balanced --model transformer --batch_size 128 --epochs 300 --lr 0.0022 --patience 10 --num_classes 47 --save_dir /scratch/gilbreth/hu953/cs57800/pth --noise_std 0.1


