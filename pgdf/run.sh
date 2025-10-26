python experiments/train_cifar_prop1_energyO_mixupX.py --preset c10.worse
python experiments/train_cifar_prop1_energyO_mixupX.py --preset c10.rand1
python experiments/train_cifar_prop1_energyO_mixupX.py --preset c10.aggre
python experiments/train_cifar_prop1_energyO_mixupX.py --preset c10.clean

python experiments/train_cifar_prop1_energyO_mixupX.py --preset c100.worse
python experiments/train_cifar_prop1_energyO_mixupX.py --preset c100.clean

python experiments/train_tinyimagenet_prop1_energy_mixupX.py --preset tinyimagenet.20sym
python experiments/train_tinyimagenet_prop1_energy_mixupX.py --preset tinyimagenet.50sym
python experiments/train_tinyimagenet_prop1_energy_mixupX.py --preset tinyimagenet.clean
python experiments/train_tinyimagenet_prop1_energy_mixupX.py --preset tinyimagenet.40asym