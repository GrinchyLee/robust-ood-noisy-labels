from warnings import filterwarnings

filterwarnings("ignore")

import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture

from torch.utils.data import DataLoader, Dataset

# import dataloader_cifar as dataloader
# import dataloader_easy 
from PreResNet import *
from preset_parser import *
import pickle
import pdb
import sys
sys.path.append('/home/young/Leeyujin/OpenOOD')
from openood.networks.resnet18_32x32 import ResNet18_32x32

import dataloader_easy_cifarn
from dataloader_cifarn import cifar_dataset
import dataloader_cifarn as dataloader

import wandb

if __name__ == "__main__":
    args = parse_args("./presets_energy_seed0_modified0.001.json")
    os.makedirs(os.path.join(args.checkpoint_path, "saved", str(args.seed)), exist_ok=True)
    logs = open(os.path.join(args.checkpoint_path, "saved", str(args.seed), "metrics.log"), "a")
    
    wandb.init(
    project="PGDF_Noisy_dataset",
    name=f"{args.dataset}_{str(args.r)}_{str(args.seed)}"  # 실험 이름
    )
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    class_num = args.num_class

    prob_trans_m = torch.zeros([class_num,class_num])
    
    
    def energy_loss(logits, temperature=1.0):
        energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
        return energy.mean()
    

    # Training
    def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, easy_trainloader):
        # estimate transition matrix
        if epoch > 150:
            global prob_trans_m
            net.eval()
            net2.eval()
            class_num = args.num_class
            temp_prob_trans_m = torch.zeros([class_num,class_num])

            with torch.no_grad():
                for (
                    batch_idx,
                    (
                        inputs_e1,
                        inputs_e2,
                        labels_e,
                    ),
                ) in enumerate(easy_trainloader):
                    inputs_e1, inputs_e2 = (
                        inputs_e1.cuda(),
                        inputs_e2.cuda(),
                    )
                    outputs_e_1 = net(inputs_e1)
                    outputs_e_2 = net(inputs_e2)

                    pe = (
                        torch.softmax(outputs_e_1, dim=1)
                        + torch.softmax(outputs_e_2, dim=1)
                    ) / 2
                    for i in range(len(labels_e)):
                        temp_prob_trans_m[labels_e[i]] += pe[i].cpu()
            eps = 1e-8
            temp_prob_trans_m = temp_prob_trans_m / (torch.sum(temp_prob_trans_m,dim=1, keepdim=True)+eps)
            temp_prob_trans_m = torch.linalg.pinv(temp_prob_trans_m).cuda()


            if not torch.isnan(temp_prob_trans_m[0][0]):
                prob_trans_m = temp_prob_trans_m.clone()

        net.train()
        net2.eval()  # fix one network and train the other

        #############################수정#########################
        try:
            unlabeled_train_iter = iter(unlabeled_trainloader)
        except:
            unlabeled_train_iter = None
        ##########################################################

        # unlabeled_train_iter = iter(unlabeled_trainloader)
        num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
        for (
            batch_idx,
            (
                inputs_x,
                inputs_x2,
                inputs_x3,
                inputs_x4,
                labels_x,
                w_x,
            ),
        ) in enumerate(labeled_trainloader):
            ########################################수정#######################################
            if unlabeled_train_iter is not None:
                try:
                    inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u = next(unlabeled_train_iter)
                except:
                    unlabeled_train_iter = iter(unlabeled_trainloader)
                    inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u = next(unlabeled_train_iter)
            else:
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u = None, None, None, None, None
            ###################################################################################

            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, args.num_class).scatter_(
                1, labels_x.view(-1, 1), 1
            )
            w_x = w_x.view(-1, 1).type(torch.FloatTensor)

            inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = (
                inputs_x.cuda(),
                inputs_x2.cuda(),
                inputs_x3.cuda(),
                inputs_x4.cuda(),
                labels_x.cuda(),
                w_x.cuda(),
            )

            if unlabeled_train_iter is not None:
                inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u = (
                    inputs_u.cuda(),
                    inputs_u2.cuda(),
                    inputs_u3.cuda(),
                    inputs_u4.cuda(),
                    labels_u.cuda(),
                )

            if unlabeled_trainloader is not None:
                with torch.no_grad():
                    # label co-guessing of unlabeled samples
                    outputs_u_1 = net(inputs_u3)
                    outputs_u_2 = net(inputs_u4)
                    outputs_u_3 = net2(inputs_u3)
                    outputs_u_4 = net2(inputs_u4)

                    pu = (
                        torch.softmax(outputs_u_1, dim=1)
                        + torch.softmax(outputs_u_2, dim=1)
                        + torch.softmax(outputs_u_3, dim=1)
                        + torch.softmax(outputs_u_4, dim=1)
                    ) / 4
                    if epoch > 150:
                        pu = pu.cuda()
                        prob_trans_m = prob_trans_m.cuda()
                        pu = torch.mm(pu,prob_trans_m) # pseudo-labels denoising
                        # pu[pu>1]=1
                        pu[pu<0]=0

                    ptu = pu

                    eps = 1e-8
                    targets_u = ptu / (ptu.sum(dim=1, keepdim=True) +eps) # normalize
                    targets_u = targets_u.detach()
            else:
                targets_u = None

            with torch.no_grad():
                # label refinement of labeled samples
                outputs_x_1 = net(inputs_x3)
                outputs_x_2 = net(inputs_x4)

                px = (
                    torch.softmax(outputs_x_1, dim=1)
                    + torch.softmax(outputs_x_2, dim=1)) / 2
                if epoch > 150:
                    # px[px>1]=1
                    px = px.cuda()
                    prob_trans_m = prob_trans_m.cuda()
                    px = torch.mm(px,prob_trans_m) # pseudo-labels denoising
                    px[px<0] = 0

                px = w_x * labels_x + (1 - w_x) * px
                ptx = px
                eps = 1e-8
                targets_x = ptx / (ptx.sum(dim=1, keepdim=True)+eps)  # normalize
                targets_x = targets_x.detach()


            # mixmatch
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)

            if unlabeled_trainloader is not None:
                all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
                all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
                w_hard = torch.cat([w_x,w_x])
            else:
                all_inputs = torch.cat([inputs_x, inputs_x2], dim=0)
                all_targets = torch.cat([targets_x, targets_x], dim=0)
                w_hard = torch.cat([w_x,w_x])

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits = net(mixed_input)

            if unlabeled_trainloader is not None:
                logits_x = logits[: batch_size * 2]
                logits_u = logits[batch_size * 2 :]
            else:
                logits_x = logits
                logits_u = None
            
            if logits_u is not None:
                Lx, Lu, lamb = criterion(
                    logits_x,
                    mixed_target[: batch_size * 2],
                    logits_u,
                    mixed_target[batch_size * 2 :],
                    epoch + batch_idx / num_iter,
                    args.warm_up,
                    w_hard,
                    epoch,
                )

                loss = Lx + lamb * Lu
            else:
                Lx,Lu,lamb = criterion(
                    logits_x,
                    mixed_target,
                    None,
                    None,
                    epoch + batch_idx / num_iter,
                    args.warm_up,
                    w_hard,
                    epoch,
                )
                loss = Lx
            # regularization
            prior = torch.ones(args.num_class) / args.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            eps = 1e-8
            penalty = torch.sum(prior * torch.log(prior / (pred_mean+eps)))

            # loss = Lx + lamb * Lu + penalty
            loss += penalty
            
            #######################################################################
            energy_loss_val = energy_loss(logits_x)
            loss = loss +0.001 * energy_loss_val
            #######################################################################
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def warmup(epoch, net, optimizer, dataloader):
        net.train()
        num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = CEloss(outputs, labels)
            if (
                args.noise_mode == "asym"
            ):  # penalize confident prediction for asymmetric noise
                penalty = conf_penalty(outputs)
                L = loss + penalty
            elif args.noise_mode == "sym":
                L = loss
            else:
                L = loss
            L.backward()
            optimizer.step()

    def test(epoch, net1, net2, size_l1, size_u1, size_l2, size_u2):
        global logs
        net1.eval()
        net2.eval()
        all_targets = []
        all_predicted = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                all_targets += targets.tolist()
                all_predicted += predicted.tolist()

        accuracy = accuracy_score(all_targets, all_predicted)
        precision = precision_score(all_targets, all_predicted, average="weighted")
        recall = recall_score(all_targets, all_predicted, average="weighted")
        f1 = f1_score(all_targets, all_predicted, average="weighted")
        results = "Test Epoch: %d, Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f, L_1: %d, U_1: %d, L_2: %d, U_2: %d" % (
            epoch,
            accuracy * 100,
            precision * 100,
            recall * 100,
            f1 * 100,
            size_l1,
            size_u1,
            size_l2,
            size_u2,
        )
        print("\n" + results + "\n")
        logs.write(results + "\n")
        logs.flush()
        return accuracy

    def eval_train(model, all_loss):
        model.eval()
        losses = torch.zeros(len(eval_loader.dataset))
        with torch.no_grad():
            for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = CE(outputs, targets)
                for b in range(inputs.size(0)):
                    losses[index[b]] = loss[b]
        eps = 1e-8
        losses = (losses - losses.min()) / (losses.max() - losses.min()+eps)
        all_loss.append(losses)

        if (
            args.average_loss > 0
        ):  # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(all_loss)
            input_loss = history[-args.average_loss :].mean(0)
            input_loss = input_loss.reshape(-1, 1)
        else:
            input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        return prob, all_loss

    def linear_rampup(current, warm_up, rampup_length=16):
        current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
        return args.lambda_u * float(current)

    class SemiLoss(object):
        def __call__(
            self, outputs_x_1, targets_x, outputs_u, targets_u, epoch, warm_up, w_hard, actual_epoch
        ):    
            if outputs_u is not None:
                probs_u = torch.softmax(outputs_u, dim=1)
                if actual_epoch >100: 
                    Lx = -torch.mean(
                        torch.sum(F.log_softmax(outputs_x_1, dim=1) * targets_x / (w_hard ** args.mt), dim=1)
                    )
                else:
                    Lx = -torch.mean(
                        torch.sum(F.log_softmax(outputs_x_1, dim=1) * targets_x , dim=1)
                    )
                Lu = torch.mean((probs_u - targets_u) ** 2)
            else:
                if actual_epoch >100:
                    Lx = -torch.mean(
                        torch.sum(F.log_softmax(outputs_x_1, dim=1) * targets_x / (w_hard ** args.mt), dim=1)
                    )
                else:
                    Lx = -torch.mean(
                        torch.sum(F.log_softmax(outputs_x_1, dim=1) * targets_x , dim=1)
                    )
                Lu = torch.tensor(0.0, device=outputs_x_1.device)
            return Lx, Lu, linear_rampup(epoch, warm_up)
        

    class NegEntropy(object):
        def __call__(self, outputs):
            probs = torch.softmax(outputs, dim=1)
            return torch.mean(torch.sum(probs.log() * probs, dim=1))

    def create_model(devices=[0]):
        model = ResNet18_32x32(num_classes=args.num_class)
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=devices).cuda()
        return model

    loader = dataloader.cifar_dataloader(
        dataset=args.dataset,
        r=args.r,
        noise_mode=args.noise_mode,
        batch_size=args.batch_size,
        warmup_batch_size=args.warmup_batch_size,
        num_workers=args.num_workers,
        root_dir=args.data_path,
        noise_file=f"{args.checkpoint_path}/saved/labels.json",
        preaug_file=(
            f"{args.checkpoint_path}/saved/{args.preset}_preaugdata.pth.tar"
            if args.preaugment
            else ""
        ),
        augmentation_strategy=args,
    )

    loader_easy = dataloader_easy_cifarn.easy_dataloader(
        dataset=args.dataset,
        r=args.r,
        noise_mode=args.noise_mode,
        batch_size=args.batch_size,
        warmup_batch_size=args.warmup_batch_size,
        num_workers=args.num_workers,
        root_dir=args.data_path,
        noise_file=f"{args.checkpoint_path}/saved/{args.seed}/easy_labels.p",
        preaug_file=(
            f"{args.checkpoint_path}/saved/{args.preset}_preaugdata.pth.tar"
            if args.preaugment
            else ""
        ),
        augmentation_strategy=args,
    )

    

    print("| Building net")
    devices = range(torch.cuda.device_count())
    net1 = create_model(devices)
    net2 = create_model(devices)

    cudnn.benchmark = True

    criterion = SemiLoss()
    optimizer1 = optim.SGD(
        net1.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    )
    optimizer2 = optim.SGD(
        net2.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    )

    all_loss = [[], []]  

    epoch = 0

    CE = nn.CrossEntropyLoss(reduction="none")
    CEloss = nn.CrossEntropyLoss()
    if args.noise_mode == "asym":
        conf_penalty = NegEntropy()

    warmup_trainloader = loader.run("warmup")
    with open(f"{args.checkpoint_path}/saved/{args.seed}/train_data_easy.p","rb") as f1:
        train_data = pickle.load(f1)
    with open(f"{args.checkpoint_path}/saved/{args.seed}/train_label_easy.p","rb") as f2:
        train_label = pickle.load(f2)
    easy_trainloader = loader_easy.run("clean",train_data,train_label)
    test_loader = loader.run("test")
    eval_loader = loader.run("eval_train")

    prob_his1 = pickle.load(open(f"{args.checkpoint_path}/saved/{args.seed}/prob1_ehn.p","rb"))
    prob_his2 = pickle.load(open(f"{args.checkpoint_path}/saved/{args.seed}/prob2_ehn.p","rb"))
    
    best_acc = 0.0  
    
    while epoch < args.num_epochs:
        lr = args.learning_rate
        if epoch >= args.lr_switch_epoch:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group["lr"] = lr
        for param_group in optimizer2.param_groups:
            param_group["lr"] = lr

        size_l1, size_u1, size_l2, size_u2 = (
            len(warmup_trainloader.dataset),
            0,
            len(warmup_trainloader.dataset),
            0,
        )

        if epoch < args.warm_up:
            print("Warmup Net1")
            warmup(epoch, net1, optimizer1, warmup_trainloader)
            print("\nWarmup Net2")
            warmup(epoch, net2, optimizer2, warmup_trainloader)

        else:
            if epoch > 200:
                prob1_gmm, all_loss[0] = eval_train(net1, all_loss[0])
                prob2_gmm, all_loss[1] = eval_train(net2, all_loss[1])

            m = args.md

            if epoch > 200:
                prob1 = m*prob1_gmm + (1-m)*prob_his1 
                prob2 = m*prob2_gmm + (1-m)*prob_his2 
            else:
                prob1 = prob_his1
                prob2 = prob_his2


            pred1 = prob1 > 0.5
            pred2 = prob2 > 0.5

            print("Train Net1")
            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred2, prob2
            )  # co-divide
            if unlabeled_trainloader is not None:
                size_l1, size_u1 = (
                    len(labeled_trainloader.dataset),
                    len(unlabeled_trainloader.dataset),
                )
            else:
                size_l1, size_u1 = (
                    len(labeled_trainloader.dataset),
                    0,
                )
            train(
                epoch,
                net1,
                net2,
                optimizer1,
                labeled_trainloader,
                unlabeled_trainloader,
                easy_trainloader,
            )  # train net1

            print("\nTrain Net2")
            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred1, prob1
            )  # co-divide
            if unlabeled_trainloader is not None:
                size_l2, size_u2 = (
                    len(labeled_trainloader.dataset),
                    len(unlabeled_trainloader.dataset),
                )
            else:
                size_l2, size_u2 = (
                    len(labeled_trainloader.dataset),
                    0,
                )
            train(
                epoch,
                net2,
                net1,
                optimizer2,
                labeled_trainloader,
                unlabeled_trainloader,
                easy_trainloader,
            )  # train net2

        acc = test(epoch, net1, net2, size_l1, size_u1, size_l2, size_u2)
        data_dict = {
            "epoch": epoch,
            "net1": net1.state_dict(),
            "net2": net2.state_dict(),
            "optimizer1": optimizer1.state_dict(),
            "optimizer2": optimizer2.state_dict(),
            "all_loss": all_loss,
        }
        wandb.log({"epoch": epoch, "loss": all_loss, "test_accuracy": acc})
        if (epoch + 1) % args.save_every == 0 or epoch == args.warm_up - 1:
            checkpoint_model = os.path.join(
                args.checkpoint_path, "all", f"{str(args.seed)}/{args.preset}_epoch{epoch}.pth.tar"
            )
            torch.save(data_dict, checkpoint_model)
        saved_model = os.path.join(
            args.checkpoint_path, "saved", str(args.seed), f"{args.preset}.pth.tar"
        )
        torch.save(data_dict, saved_model)

        if epoch ==150:
            saved_model_150 = os.path.join(
            args.checkpoint_path, "saved", str(args.seed), f"{args.preset}_ep150_energy_tsX.pth.tar"
            )
        if epoch ==200 :
            saved_model_200 = os.path.join(
            args.checkpoint_path, "saved", str(args.seed), f"{args.preset}_ep200_energy_tsX.pth.tar"
            )
            torch.save(data_dict, saved_model_200)
        if epoch ==250 :
            saved_model_250 = os.path.join(
            args.checkpoint_path, "saved", str(args.seed), f"{args.preset}_ep250_energy_tsX.pth.tar"
            )
            torch.save(data_dict, saved_model_200)
        if epoch ==300 :
            saved_model_300 = os.path.join(
            args.checkpoint_path, "saved", str(args.seed), f"{args.preset}_ep300_energy_tsX.pth.tar"
            )
            torch.save(data_dict, saved_model_300)
        epoch += 1          