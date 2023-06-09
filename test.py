import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import ttach as tta

GPUS = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPUS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
    
def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(42)

transform_test = A.Compose([
        A.Resize(224,224),
        ToTensorV2()
])

testset = Cifar10SearchDataset(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

models = []
model_path = './models'
m_num = 0
m_total = 0

model_name = 'resnet18'
for model_parameter in sorted(os.listdir(model_path)):
    # 특정 test accuracy 이상인 model들만 불러오기
    m_total += 1
    if int(model_parameter[-8:-4]) < 9600:
        continue
    m_num += 1
    model = timm.create_model(model_name, num_classes=10)
    saved_checkpoint = torch.load(os.path.join(model_path, model_parameter))
    model.load_state_dict(saved_checkpoint, strict=False)
    model.eval()
    model = model.to(device)
    models.append(model)


# tta (test time augmentation)
tta_transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        # tta.Add([1,2]),
        # tta.Multiply([1, 1.05, 1.1])
        # tta.Scale(scales=[1, 2])
        # tta.FiveCrops(200, 200)
    ]
)

correct = 0
total = 0
seed_all(42)
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device).float(), labels.to(device)
        outputs = torch.zeros(100, 10).to(device)
        for model in models:
            tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)
            model_output = tta_model(images)
            outputs += model_output
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the ensemble on the 10000 test images: {(100 * correct / total)}')
print(f'특정 점수 넘는 모델 개수: {m_num} / {m_total}')
