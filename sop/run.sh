python train_cifar.py -c config_cifar10N.json --lr_u 10 --lr_v 10 --percent 0.0 --seed 0
python train_cifar.py -c config_cifar10N.json --lr_u 10 --lr_v 10 --percent 9.0 --seed 0
python train_cifar.py -c config_cifar10N.json --lr_u 10 --lr_v 10 --percent 17.2 --seed 0
python train_cifar.py -c config_cifar10N.json --lr_u 10 --lr_v 10 --percent 40.2 --seed 0

python train_cifar.py -c config_cifar100N.json --lr_u 1 --lr_v 5 --percent 0.0 --seed 0
python train_cifar.py -c config_cifar100N.json --lr_u 1 --lr_v 5 --percent 40.2 --seed 0

python train_TIN.py -c config_TIN.json --lr_u 1 --lr_v 5 --percent 20.0 --seed 0 --noise_file "OpenOOD/data/benchmark_imglist/tinyimagenet/train_tin_sym_0.2.txt"
python train_TIN.py -c config_TIN.json --lr_u 1 --lr_v 5 --percent 50.0 --seed 0 --noise_file "OpenOOD/data/benchmark_imglist/tinyimagenet/train_tin_sym_0.5.txt"
python train_TIN.py -c config_TIN.json --lr_u 1 --lr_v 5 --percent 0.0 --seed 0 --noise_file "OpenOOD/data/benchmark_imglist/tinyimagenet/train_tin.txt"
python train_TIN.py -c config_TIN.json --lr_u 1 --lr_v 5 --percent 40.0 --seed 0 --noise_file "OpenOOD/data/benchmark_imglist/tinyimagenet/train_tin_asym_0.4.txt"