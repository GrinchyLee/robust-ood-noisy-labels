python train_cifar.py -c config_cifar10N_cosinewarming_seed0.json --seed 0 --beta 0.7 --lamb 3 --percent 9.0
python train_cifar.py -c config_cifar10N_cosinewarming_seed0.json --seed 0 --beta 0.7 --lamb 3 --percent 0
python train_cifar.py -c config_cifar10N_cosinewarming_seed0.json --seed 0 --beta 0.7 --lamb 3 --percent 17.2
python train_cifar.py -c config_cifar10N_cosinewarming_seed0.json --seed 0 --beta 0.7 --lamb 3 --percent 40.2

python train_cifar.py -c config_cifar100N_cosinewarming_seed0.json --seed 0 --beta 0.9 --lamb 7 --percent 0
python train_cifar.py -c config_cifar100N_cosinewarming_seed0.json --seed 0 --beta 0.9 --lamb 7 --percent 40.2

python train_TIN.py -c config_TIN_seed0.json --seed 0 --beta 0.9 --lamb 7 --percent 20.0 --noise_file "train_tin_sym_0.2.txt"
python train_TIN.py -c config_TIN_seed0.json --seed 0 --beta 0.9 --lamb 7 --percent 0.0 --noise_file "train_tin.txt"
python train_TIN.py -c config_TIN_seed0.json --seed 0 --beta 0.9 --lamb 7 --percent 40.0 --noise_file "train_tin_asym_0.4.txt"
python train_TIN.py -c config_TIN_seed0.json --seed 0 --beta 0.9 --lamb 7 --percent 50.0 --noise_file "train_tin_sym_0.5.txt"