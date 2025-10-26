#!/bin/bash
# sh scripts/osr/opengan/cifar100_train_opengan.sh

SEED=0

# feature extraction
python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
            configs/datasets/cifar100/cifar100_ood.yml \
            configs/networks/resnet18_32x32.yml \
            configs/pipelines/train/train_opengan_feat_extract.yml \
            configs/preprocessors/base_preprocessor.yml \
    --network.checkpoint "/home/yujin/ProMix2/save/seed0/no_mixup_v2_entropyy0.01_cifar100_noisy_label_0_0.5,0.5_300_epoch_300_model_net1.pth.tar" \
    --seed 0

# train
python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
            configs/networks/opengan.yml \
            configs/pipelines/train/train_opengan.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/opengan.yml \
    --dataset.feat_root ./results/cifar100_resnet18_32x32_feat_extract_opengan_default/s0 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint /home/yujin/ProMix2/save/seed0/no_mixup_v2_entropyy0.01_cifar100_noisy_label_0_0.5,0.5_300_epoch_300_model_net1.pth.tar \
    --seed 0
