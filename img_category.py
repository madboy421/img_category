import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import os

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
import torchvision.transforms as transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def create_dataset():
    # 1. 加载数据集
    def unpickle(file):
        with open(file, 'rb') as fo:    # 二进制方式读
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # 1.1 加载训练集
    batch_data = unpickle('../data/cifar-10-python/cifar-10-batches-py/data_batch_1')
    X_train = torch.Tensor(batch_data[b'data']).reshape(-1, 3, 32, 32)
    y_train = torch.tensor(batch_data[b'labels'], dtype=torch.int64)

    for i in range(1, 5):
        batch_data = unpickle(f'../data/cifar-10-python/cifar-10-batches-py/data_batch_{i + 1}')
        img = torch.Tensor(batch_data[b'data']).reshape(-1, 3, 32, 32)
        labels = torch.tensor(batch_data[b'labels'], dtype=torch.int64)
        X_train = torch.cat((X_train, img), dim=0)
        y_train = torch.cat((y_train, labels))

    # 1.2 加载测试集
    test_batch_data = unpickle(f'../data/cifar-10-python/cifar-10-batches-py/test_batch')
    X_test = torch.Tensor(test_batch_data[b'data']).reshape(-1, 3, 32, 32)
    y_test = torch.tensor(test_batch_data[b'labels'], dtype=torch.int64)

    # 数据标准化 (使用CIFAR-10的均值和标准差)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)  # 转换维度
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

    X_train = (X_train / 255.0 - mean) / std
    X_test = (X_test / 255.0 - mean) / std

    # 构建TensorDataset
    train_tensordataset = TensorDataset(X_train, y_train)
    test_tensordataset = TensorDataset(X_test, y_test)

    return train_tensordataset, test_tensordataset


# 改进的模型架构——参考VGG16网络: 采用小卷积核，多个小卷积核作用等同于一个大的卷积核，并且参数量更小的同时还能引入更多的激活函数。同时卷积层不改变尺寸，仅在池化层改变
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)   # 等同于nn.Flatten()
        x = self.classifier(x)
        return x

# 数据增强函数
class CIFAR10Augmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    def __call__(self, img):
        # 将Tensor转换为PIL图像，然后再转换回Tensor，虽然效率低，但可以接受
        #   将Tensor转换为numpy进行增强
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()  # PIL图像通道数位于最后

        #   简单的数据增强实现
        img = torch.tensor(img).permute(2, 0, 1).float()

        # 随机裁剪
        if torch.rand(1) > 0.5:
            pad = 4
            padded = F.pad(img.unsqueeze(0), (pad, pad, pad, pad), mode='reflect').squeeze(0)
            h, w = 32, 32
            top = torch.randint(0, pad * 2, (1,))
            left = torch.randint(0, pad * 2, (1,))
            img = padded[:, top:top + h, left:left + w]

        # 随机水平翻转
        if torch.rand(1) > 0.5:
            img = torch.flip(img, [2])

        return img


# 训练和测试函数
def train_test(model, train_tensorDataset, test_tensorDataset, lr, epoch_num, batch_size, device):
    def init_weights(layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    model.apply(init_weights)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 使用AdamW和权重衰减

    # 学习率调度器
    # 模拟余弦函数的方式，在训练过程中动态调整学习率，以帮助模型更好地收敛，
    # - T_max: 余弦半个周期的长度，lr将在T_max轮训练从最初初始化的值->eta_min 它定义了学习率完成半个余弦周期所经历的epoch数
    # - eta_min lr在训练完T_max轮数后应等于eta_min，默认为0
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

    # 数据增强
    augmentation = CIFAR10Augmentation()

    train_loader = DataLoader(train_tensorDataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensorDataset, batch_size=batch_size, shuffle=False)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    best_test_acc = 0.0

    for epoch in range(epoch_num):
        model.train()
        total_train_loss = 0
        total_train_acc_num = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            # 应用数据增强 (只在训练时)
            if model.training:
                X_aug = torch.stack([augmentation(x) for x in X])
                X = X_aug.to(device)

            optimizer.zero_grad()
            output = model(X)
            loss_val = loss_fn(output, y)
            loss_val.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += loss_val.item() * X.size(0)
            total_train_acc_num += torch.sum(y == torch.argmax(output, dim=1)).item()

            print(f"\rEpoch {epoch + 1:0>2} [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss_val.item():.4f}", end="")

        scheduler.step()

        # 计算训练准确率
        this_train_loss = total_train_loss / len(train_tensorDataset)
        this_train_acc = total_train_acc_num / len(train_tensorDataset)
        train_loss_list.append(this_train_loss)
        train_acc_list.append(this_train_acc)

        # 测试
        model.eval()
        total_test_acc_num = 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                total_test_acc_num += torch.sum(y == torch.argmax(output, dim=1)).item()

        this_test_acc = total_test_acc_num / len(test_tensorDataset)
        test_acc_list.append(this_test_acc)

        # 保存最佳模型
        if this_test_acc > best_test_acc:
            best_test_acc = this_test_acc
            torch.save(model.state_dict(), 'best_cifar10_model.pth')

        print(f" | Train Loss: {this_train_loss:.4f}, Train Acc: {this_train_acc:.4f}, "
              f"Test Acc: {this_test_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"Best Test Accuracy: {best_test_acc:.4f}")
    return train_loss_list, train_acc_list, test_acc_list


# 测试
train_dataset, test_dataset = create_dataset()

# 初始化超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001  # 降低学习率
epoch_num = 30  # 增加训练轮数
batch_size = 128  # 调整批量大小

# 创建改进的模型
model = ImprovedCNN()

print(f"Using device: {device}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

train_loss_list, train_acc_list, test_acc_list = train_test(
    model, train_dataset, test_dataset, lr, epoch_num, batch_size, device
)

# 加载最佳模型进行最终测试
model.load_state_dict(torch.load('best_cifar10_model.pth'))
model.eval()

# 测试单张图片
img, label = test_dataset[1306]
with torch.no_grad():
    output = model(img.unsqueeze(0).to(device))
    pred = torch.argmax(output, dim=1).item()

print(f"Prediction: {pred}, True Label: {label.item()}")
print(f"Correct: {pred == label.item()}")

# 打印参数量
print("----------------打印参数-------------")
for i in model.state_dict():
    print(f"{i}: {model.state_dict()[i]}")
print()

# 绘制结果
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(((img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) +
               torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)) * 255).permute(1, 2, 0).int())
ax[0].axis('off')
ax[0].set_title('Test Image')

ax[1].plot(train_acc_list, 'r-', label='train_accuracy', alpha=0.7)
ax[1].plot(test_acc_list, 'b--', label='test_accuracy', alpha=0.7)
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Training and Test Accuracy')
ax[1].legend(loc='lower right')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 绘制损失曲线
plt.figure(figsize=(8, 5))
plt.plot(train_loss_list, 'g-', label='train_loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()