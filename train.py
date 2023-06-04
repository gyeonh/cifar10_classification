#################
#### Library ####
#################
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms

import random
import math
import time
import datetime
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import ttach as tta


##############
#### seed ####
##############
def seed_all(seed): 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#############
#### GPU ####
#############
MultiGPUS = True
GPUS = '2, 3'
os.environ['CUDA_VISIBLE_DEVICES'] = GPUS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#########################
#### Hyperparameters ####
#########################
seed_number = 1234   # 42
epochs = 100
lr = 0.01   # 0


#################
#### Dataset ####
#################
class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

transform_test = A.Compose([
        A.Resize(224,224),
        ToTensorV2()
])

testset = Cifar10SearchDataset(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)


######################
#### lr_scheduler ####
######################
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            

################################################
#### conditional list (except lr_scheduler) ####
################################################
augmentation_list = [
            A.Compose([
                            A.Resize(224,224),
                            A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),     # 밝기와 대비 변경 (대비 올리면 어두운 색을 더 어둡게, 밝은색을 더 밝게)
                            A.RandomGamma(gamma_limit=(90, 110)),
                            A.OneOf([A.NoOp(), A.MultiplicativeNoise(), A.GaussNoise(), A.ISONoise()]),
                            A.OneOf(
                                [
                                    A.NoOp(p=0.8),
                                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10),
                                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10)
                                ],
                                p=0.2,
                            ),
                            A.OneOf([A.ElasticTransform(), A.GridDistortion(), A.NoOp()]),
                            ToTensorV2(),
                        ]),
            A.Compose([
                            A.Resize(224,224),
                            A.HorizontalFlip(p=0.5),
                            A.OneOf([A.NoOp(), A.MultiplicativeNoise(), A.GaussNoise(), A.ISONoise()]),
                            A.OneOf(
                                [
                                    A.NoOp(p=0.8),
                                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10),
                                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10)
                                ],
                                p=0.2,
                            ),
                            A.OneOf([A.ElasticTransform(), A.GridDistortion(), A.NoOp()]),
                            ToTensorV2(),
                        ])
                    ]
optimizer_list = ['AdamW', 'SGD_weight_decay']  # additional optimizer candidate in 1t -> 'SGD'
batch_size_list = [64, 128]


###############################
#### find best combination ####
###############################
model_name = 'resnet18'
for aug_idx, transform_train in enumerate(augmentation_list):
    for opt_idx, optimizer_type in enumerate(optimizer_list):
        for bat_idx, batch_size in enumerate(batch_size_list):
            #### seed
            seed_all(seed_number)
            
            #### model
            model = timm.create_model(model_name, pretrained=True, num_classes=10)
            
            if MultiGPUS:
                model = nn.DataParallel(model).to(device)
            else:
                model = model.to(device)

            trainset = Cifar10SearchDataset(root='./data', train=True, download=False, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)
            
            criterion = nn.CrossEntropyLoss()
            
            #### optimizer
            if optimizer_type == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            elif optimizer_type == 'SGD':   # only for 1t
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            elif optimizer_type == 'SGD_weight_decay':
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            
            #### lr_scheduler
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=5, T_mult=1, eta_max=0.01, T_up=10, gamma=0.5)

            #### train
            start_time = time.time()
            for epoch in range(epochs):
                temp_time = time.time()
                model.train()
                running_loss = 0.0
                train_total = 0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device).float(), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    train_total += labels.size(0)
                    
                #### evaluation
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        images, labels = images.to(device).float(), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                scheduler.step()
                expected_time = (time.time()-temp_time)*(epochs-epoch) + (time.time()-start_time)
                print(f"[{epoch + 1:03d}/{epochs}], Loss: {running_loss / train_total:4.3f}, Test accuracy: {100 * correct / total:4.2f} % --- \
{str(datetime.timedelta(seconds=time.time()-start_time)).split('.')[0]} / {str(datetime.timedelta(seconds=expected_time)).split('.')[0]}")
            test_acc = 100 * correct / total
            print(f'Finished Training\nLast Accuracy of {aug_idx}_{opt_idx}_{batch_size:03d}: {test_acc:4.2f}!!')
            
            #### save weights
            PATH = f'./models/6t_{model_name}_aug{aug_idx}_opt{opt_idx}_{batch_size:03d}_{int(test_acc*100)}.pth'
            torch.save(model.state_dict(), PATH)
