import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pickle
import os
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from torchvision import transforms
from torch.optim import lr_scheduler

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class DatasetAugmentation():
    def __init__(self, root_dir, train=True, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []

        if train:
            # 加载训练集
            file_list = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            # 加载测试集
            file_list = [f'test_batch']

        # 使用pickle加载数据集
        for file_name in file_list:
            file_path = os.path.join(root_dir, file_name)
            with open(file_path, 'rb') as f:
                dict = pickle.load(f,encoding='bytes')
            # # 将数据重塑为 (10000, 3, 32, 32) 并转换为uint8，然后调整轴顺序为PyTorch常用的(10000, 32, 32, 3)
            self.data.append(dict[b'data'].reshape(-1,3,32,32).transpose((0,2,3,1)))
            self.labels.extend(dict[b'labels'])

        # 将所有batch的标签和data合并
        self.data = np.vstack(self.data,dtype=np.float32)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取单张图像和标签
        data = self.data[idx]
        label = self.labels[idx]

        # 将numpy数组转换为PIL图像，方便transform.toTensor()
        image = Image.fromarray(data.copy().astype(np.uint8))

        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataset():
    # 1. 加载数据集
    def unpickle(file):
        with open(file, 'rb') as fo:
            # 在Python 3中需要指定编码
            dict = pickle.load(fo, encoding='bytes')
        return dict
    # 1.1 加载训练集
    # method1
    # X_train = torch.empty((1,3,32,32))
    # y_train = torch.empty((1))
    # for i in torch.arange(0,5,dtype=torch.uint8):
    #     batch_data = unpickle(f'../data/cifar-10-python/cifar-10-batches-py/data_batch_{i+1}')
    #     img = torch.Tensor(batch_data[b'data']).reshape(-1,3,32,32)
    #     labels = torch.tensor(batch_data[b'labels'], dtype=torch.int64)
    #     X_train = torch.cat((X_train, img), dim=0)
    #     y_train = torch.cat((y_train, labels))
    # # print(X_train,X_train.shape)
    # # 移除第1个未初始化的数据
    # X_train = torch.cat([X_train[1:2],X_train[2:]], dim=0)
    # print(X_train,X_train.shape)
    # method2
    batch_data = unpickle('../data/cifar-10-python/cifar-10-batches-py/data_batch_1')
    # 注意：键名是字节字符串（byte strings）
    X_train = torch.Tensor(batch_data[b'data']).reshape(-1,3,32,32)     # 形状为(10000, 3072)的numpy数组
    y_train = torch.tensor(batch_data[b'labels'], dtype=torch.int64)    # 包含10000个标签的列表
    for i in torch.arange(1,5,dtype=torch.uint8):
        batch_data = unpickle(f'../data/cifar-10-python/cifar-10-batches-py/data_batch_{i+1}')
        # 获取图像数据和标签
        # 注意：键名是字节字符串（byte strings）
        img = torch.Tensor(batch_data[b'data']).reshape(-1,3,32,32)
        labels = torch.tensor(batch_data[b'labels'], dtype=torch.int64)
        X_train = torch.cat((X_train, img), dim=0)
        y_train = torch.cat((y_train, labels))
    # 1.2 加载测试集
    test_batch_data = unpickle(f'../data/cifar-10-python/cifar-10-batches-py/test_batch')
    # 获取数据和标签
    X_test = torch.Tensor(test_batch_data[b'data']).reshape(-1,3,32,32)
    y_test = torch.tensor(test_batch_data[b'labels'],dtype=torch.int64)

    # 构建TensorDataset
    # 数据标准化 (使用CIFAR-10的均值和标准差)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)  # 转换维度
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

    X_train = (X_train / 255.0 - mean) / std
    X_test = (X_test / 255.0 - mean) / std

    train_tensordataset = TensorDataset(X_train, y_train)
    test_tensordataset = TensorDataset(X_test, y_test)

    return train_tensordataset, test_tensordataset

#     transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # CIFAR-10常用标准化参数
#     ])
#
#     train_dataset = DatasetAugmentation(root_dir='../data/cifar-10-python/cifar-10-batches-py', train=True,
#                                         transform=transform)
#     test_dataset = DatasetAugmentation(root_dir='../data/cifar-10-python/cifar-10-batches-py', train=False,
#                                        transform=transform)
#     return train_dataset, test_dataset


# 4 构建模型结构
model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=2),
    # nn.BatchNorm2d(12),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
    # nn.Dropout(0.20),
    nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5, stride=1, padding=0),
    # nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
    # nn.Dropout(0.20),
    # nn.Conv2d(in_channels=16, out_channels=20, kernel_size=3, stride=1),
    # nn.ReLU(),
    # nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=16*6*6, out_features=120),
    nn.BatchNorm1d(num_features=120),
    nn.ReLU(),
    nn.Dropout(p=0.20),
    nn.Linear(in_features=120, out_features=84),
    nn.BatchNorm1d(num_features=84),
    nn.ReLU(),
    nn.Dropout(p=0.20),
    nn.Linear(in_features=84, out_features=10),
)

# train and test
def train_test(model,train_tensorDataset, test_tensorDataset,lr, epoch_num, batch_size, device):
    # 初始化参数
    def init_weights(layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
    model.apply(init_weights)
    # ->device
    model.to(device)
    # loss
    loss = nn.CrossEntropyLoss()
    # 构建optimizer dataloader
    # 1.Adam
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 2.lr_scheduler
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler_lr = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    # 3.momentum  lr=0.001
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # 4.RMSProp  lr=0.01
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    # 4.AdaGrad  lr = 0.01
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    # 设置余弦学习率衰减
    scheduler_lr = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

    train_loader = DataLoader(train_tensorDataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensorDataset, batch_size=batch_size, shuffle=True)
    # 构造容器
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    # 开始训练
    for epoch in range(epoch_num):
        total_train_loss = 0
        total_train_acc_num = 0
        total_test_acc_num = 0
        # 进入训练模式
        model.train()
        for batch_idx,(X,y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            # forward
            output = model(X)
            # loss_val
            loss_val = loss(output, y)
            total_train_loss += loss_val.item() * X.size(0)
            # backward
            loss_val.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # update weights
            optimizer.step()
            optimizer.zero_grad()
            # acc
            total_train_acc_num += torch.sum(y == torch.argmax(output, dim=1)).item()
            # print 进度条
            # print(f"\r{epoch+1:0>2}[{'='* int((batch_idx+1)/len(train_loader) * 50):<50}]",end="")
            print(f"\repoch{epoch+1:0>2}[{(batch_idx+1)}/{len(train_loader)}]",end="")
        # lr decend
        scheduler_lr.step()
        # print loss_val and acc
        this_train_loss = total_train_loss / len(train_tensorDataset)
        this_train_acc = total_train_acc_num / len(train_tensorDataset)
        train_loss_list.append(this_train_loss)
        train_acc_list.append(this_train_acc)

        # test
        model.eval()
        with torch.no_grad():
            for X,y in test_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                total_test_acc_num += torch.sum(y == torch.argmax(output, dim=1)).item()
            # acc
            this_test_acc = total_test_acc_num / len(test_tensorDataset)
            test_acc_list.append(this_test_acc)
            # print loss acc
            print(f"train_loss: {this_train_loss:.4f}, train_acc: {this_train_acc:.4f}, test_acc: {this_test_acc:.4f}, LR: {scheduler_lr.get_last_lr()[0]}")
    return train_loss_list, train_acc_list, test_acc_list,optimizer
# 测试
train_dataset, test_dataset = create_dataset()

# 初始化超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
epoch_num = 30
batch_size = 128
#
train_loss_list, train_acc_list, test_acc_list,optimizer = train_test(model,train_dataset,test_dataset,lr,epoch_num,batch_size,device)

# 打印参数
# summary(model,(3,32,32),batch_size=60000,device=device)

# save model
# joblib.dump(model,'img_category_model')

# 传入一张图片测试
img, label = test_dataset[1306]
with torch.no_grad():
    output = model(img.unsqueeze(0).to(device))
    pred = torch.argmax(output, dim=1).item()
print(pred)
print(label.item())
print(pred == label.item())

fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(((img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) +
               torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)) * 255).permute(1, 2, 0).int())
ax[0].axis('off')
ax[0].set_title('test Image')
ax[1].plot(train_acc_list,'r-',label='train_accuracy')
ax[1].plot(test_acc_list,'b--',label='test_accuracy')
ax[1].legend(loc='lower right')
ax[2].plot(train_loss_list,'g-',label='train_loss')
ax[2].legend(loc='upper right')
fig.suptitle(f'lr={lr:.4f},epoch={epoch_num},batch_size={batch_size},optimizer={optimizer.__class__.__name__}')
plt.show()