# clean
python experiments/Train_cifar_prop1_energyO_mixupX.py --noise_type clean --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10  --noise_mode cifarn --num_epochs 300 --seed 0
# aggre
python experiments/Train_cifar_prop1_energyO_mixupX.py --noise_type aggre --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10  --noise_mode cifarn --num_epochs 300 --seed 0
# rand1
python experiments/Train_cifar_prop1_energyO_mixupX.py --noise_type rand1 --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10  --noise_mode cifarn --num_epochs 300 --seed 0
# worst
python experiments/Train_cifar_prop1_energyO_mixupX.py --noise_type worst --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10  --noise_mode cifarn --num_epochs 300 --seed 0

# noisy100
python experiments/Train_cifar_prop1_energyO_mixupX.py --noise_type noisy100 --cosine --dataset cifar100 --num_class 100 --rho_range 0.5,0.5 --tau 0.95 --pretrain_ep 30 --debias_output 0.5 --debias_pl 0.5 --noise_mode cifarn --num_epochs 300 --seed 0
# clean100
python experiments/Train_cifar_prop1_energyO_mixupX.py --noise_type clean100 --cosine --dataset cifar100 --num_class 100 --rho_range 0.5,0.5 --tau 0.95 --pretrain_ep 30 --debias_output 0.5 --debias_pl 0.5 --noise_mode cifarn --num_epochs 300 --seed 0

python Train_TIN_prop1_energyO_mixupX.py --cosine --dataset tinyimagenet --num_class 200 --rho_range 0.5,0.5 --tau 0.95 --pretrain_ep 30 --debias_output 0.5 --debias_pl 0.5 --noise_mode clean --num_epochs 300 --noise_rate 0.0 --noise_file "train_tin.txt" --seed 0 
python Train_TIN_prop1_energyO_mixupX.py --cosine --dataset tinyimagenet --num_class 200 --rho_range 0.5,0.5 --tau 0.95 --pretrain_ep 30 --debias_output 0.5 --debias_pl 0.5 --noise_mode sym0.2 --num_epochs 300 --noise_rate 0.2 --noise_file "train_tin_sym_0.2.txt" --seed 0 
python Train_TIN_prop1_energyO_mixupX.py --cosine --dataset tinyimagenet --num_class 200 --rho_range 0.5,0.5 --tau 0.95 --pretrain_ep 30 --debias_output 0.5 --debias_pl 0.5 --noise_mode sym0.5 --num_epochs 300 --noise_rate 0.5 --noise_file "train_tin_sym_0.5.txt" --seed 0 
python Train_TIN_prop1_energyO_mixupX.py --cosine --dataset tinyimagenet --num_class 200 --rho_range 0.5,0.5 --tau 0.95 --pretrain_ep 30 --debias_output 0.5 --debias_pl 0.5 --noise_mode asym0.4 --num_epochs 300 --noise_rate 0.4 --noise_file "train_tin_asym_0.4.txt" --seed 0 