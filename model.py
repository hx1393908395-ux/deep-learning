import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import time
import os


# ===================== 你写的SE-ResNet18模型 =====================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_se=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_se=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_se = use_se

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_se=self.use_se))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(use_se=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, use_se=use_se)


# ===================== 数据加载+训练+测试逻辑 =====================
# 设备配置：优先使用GPU，无GPU则用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 超参数设置（针对CIFAR-10调优）
batch_size = 128
learning_rate = 0.001
num_epochs = 20
save_path = "se_resnet18_cifar10.pth"  # 模型保存路径

# CIFAR-10数据预处理：归一化+数据增强（训练集）
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10均值/方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载CIFAR-10数据集（自动下载到./data文件夹）
train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

# 数据加载器：开启多进程（num_workers>0），解决之前的multiprocessing报错
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=True  # pin_memory=True加速GPU数据传输
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=0, pin_memory=True
)


# 定义训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 数据移到设备上
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播+优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计指标
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 打印批次信息
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {total_loss / (batch_idx + 1):.4f}, Acc: {100. * correct / total:.2f}%, '
                  f'Time: {time.time() - start_time:.2f}s')
    return total_loss / len(train_loader), 100. * correct / total


# 定义测试函数
def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算，节省内存
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100. * correct / total
    loss = total_loss / len(test_loader)
    print(f'Test Loss: {loss:.4f}, Test Acc: {acc:.2f}%')
    return loss, acc


# ===================== 主程序入口（关键：解决多进程报错） =====================
if __name__ == '__main__':
    # 1. 初始化两个模型：SE-ResNet18 和 普通ResNet18（用于对比）
    model_se = ResNet18(use_se=True).to(device)  # 带SE注意力
    model_vanilla = ResNet18(use_se=False).to(device)  # 无SE注意力（原版ResNet18）

    # 选择训练的模型：注释掉另一行即可切换
    model = model_se
    # model = model_vanilla

    # 2. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 分类任务用交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器（比SGD更稳定）
    # 学习率调度器：每5个epoch学习率减半（可选，提升精度）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 3. 训练过程
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0.0  # 保存最优模型的测试精度

    print("开始训练...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = test(model, test_loader, criterion)
        scheduler.step()  # 更新学习率

        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"保存最优模型，当前最高测试精度: {best_acc:.2f}%")

    # 4. 训练完成后加载最优模型
    model.load_state_dict(torch.load(save_path))
    final_test_loss, final_test_acc = test(model, test_loader, criterion)
    print(f"训练完成！最优模型测试精度: {final_test_acc:.2f}%")

    # 5. 绘制训练曲线（损失+精度）
    plt.figure(figsize=(12, 4))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 精度曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('se_resnet18_train_curve.png')
    plt.show()

    # CIFAR-10类别标签
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # 测试单张图片（可选）
    def predict_single_image(model, image_path, transform):
        from PIL import Image
        model.eval()
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            _, pred = output.max(1)
        return classes[pred.item()]
    # # 取消注释可测试单张图片（替换为你的图片路径）
    # print(predict_single_image(model, "test_img.jpg", transform_test))