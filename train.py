import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import time
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
from model import ResNet18


# ==========================================
# 补全utils.py的函数（如果没有utils.py，直接写在这里，避免导入错误）
# ==========================================
def count_parameters(model):
    """计算模型总参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_history(train_losses, test_accs, save_dir, model_name):
    """绘制训练曲线，解决中文显示问题"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='训练损失')
    ax1.set_title(f'{model_name} 训练损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(range(1, len(test_accs) + 1), test_accs, 'r-', label='测试精度')
    ax2.set_title(f'{model_name} 测试精度曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 参数设置（默认批次64，避免显存溢出）
# ==========================================
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Course Design')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--batch-size', default=64, type=int, help='batch size')  # 改64避显存溢出
parser.add_argument('--no-se', action='store_true', help='使用普通ResNet，不加SE')
parser.add_argument('--save-dir', default='./results', type=str, help='结果保存目录')
args = parser.parse_args()

# ==========================================
# 设备配置：CUDA优先 → MPS → CPU（CUDA加速）
# ==========================================
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
if device == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
print(f"==> Using Device: {device}")

# ==========================================
# 1. 数据准备（Windows专属：num_workers=0，避免多进程报错）
# ==========================================
print('==> Preparing CIFAR-10 Data...')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                          pin_memory=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)

# ==========================================
# 2. 模型构建
# ==========================================
use_se = not args.no_se
model_name = "SE-ResNet18" if use_se else "ResNet18-Baseline"
print(f'==> Building Model: {model_name}')
net = ResNet18(use_se=use_se).to(device)
params = count_parameters(net)
print(f"==> Total Parameters: {params / 1e6:.2f}M")

# ==========================================
# 3. 优化器与损失（损失函数移GPU）
# ==========================================
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
scaler = GradScaler() if device == 'cuda' else None  # 混合精度缩放器

# ==========================================
# 4. 训练循环（混合精度训练）
# ==========================================
train_losses = []
test_accs = []
best_acc = 0
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
log_file = open(os.path.join(args.save_dir, f'{model_name}_log.txt'), 'w')


def log_print(text):
    print(text)
    log_file.write(text + '\n')
    log_file.flush()


start_time = time.time()

for epoch in range(args.epochs):
    net.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # 混合精度训练（GPU专用）
        if device == 'cuda' and scaler is not None:
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(trainloader)
    train_losses.append(epoch_loss)

    # 测试阶段
    net.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total_test += targets.size(0)
            correct_test += predicted.eq(targets).sum().item()

    acc = 100. * correct_test / total_test
    test_accs.append(acc)
    # 保存最佳模型
    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), os.path.join(args.save_dir, f'{model_name}_best.pth'))
        save_msg = " [Saved Best]"
    else:
        save_msg = ""

    log_print(f'Epoch [{epoch + 1}/{args.epochs}] '
              f'Loss: {epoch_loss:.4f} | '
              f'Test Acc: {acc:.2f}% | '
              f'LR: {optimizer.param_groups[0]["lr"]:.4f}{save_msg}')
    scheduler.step()

# 训练完成
total_time = time.time() - start_time
log_print(f"\nTraining Finished in {total_time / 60:.1f} minutes.")
log_print(f"Best Accuracy: {best_acc:.2f}%")
log_file.close()

# 绘制训练曲线
print("==> Generating plots...")
plot_history(train_losses, test_accs, args.save_dir, model_name)
print("==> Done! All results saved in ./results")