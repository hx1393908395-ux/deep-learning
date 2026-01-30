import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
from model import ResNet18  # 从你的model.py导入ResNet18

# ==========================================
# 命令行参数设置（灵活切换测试模式）
# ==========================================
parser = argparse.ArgumentParser(description='Test SE-ResNet18/ResNet18 on CIFAR-10')
parser.add_argument('--no-se', action='store_true', help='测试普通ResNet18，不加SE模块（和训练时一致）')
parser.add_argument('--img-path', type=str, default=None, help='单张测试图片的路径，如./test_img.jpg')
parser.add_argument('--model-path', type=str, default=None, help='模型权重文件路径，默认自动从./results加载')
parser.add_argument('--save-dir', type=str, default='./results', help='训练结果保存目录，用于自动加载模型')
args = parser.parse_args()

# ==========================================
# 1. 基础配置：设备/类别标签/预处理（和train.py完全一致！）
# ==========================================
# 设备配置：CUDA优先 → MPS → CPU（和train.py设备逻辑完全一致）
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print(f"==> Using Device: {device}")

# CIFAR-10 类别标签（和数据集完全对应）
cifar10_classes = [
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 预处理变换：和train.py的测试集预处理完全一致（仅ToTensor+归一化，无数据增强）
# 单张图片/测试集通用，避免重复定义
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 强制缩放到32×32（CIFAR-10标准尺寸）
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 复用train.py的归一化参数
])

# ==========================================
# 2. 模型加载：自动加载/手动指定（和训练时的模型结构一致）
# ==========================================
use_se = not args.no_se
model_name = "SE-ResNet18" if use_se else "ResNet18-Baseline"
default_model_path = os.path.join(args.save_dir, f'{model_name}_best.pth')
model_weight_path = args.model_path if args.model_path is not None else default_model_path

# 检查模型权重文件是否存在
if not os.path.exists(model_weight_path):
    raise FileNotFoundError(
        f"模型权重文件不存在：{model_weight_path}\n请先运行train.py训练模型，或检查--no-se参数是否和训练时一致")

# 初始化模型并加载权重
print(f"==> Loading Model: {model_name}")
net = ResNet18(use_se=use_se).to(device)
# 加载权重（兼容GPU/CPU加载，自动映射设备）
net.load_state_dict(torch.load(model_weight_path, map_location=device))
net.eval()  # 关键：评估模式，关闭BN/Dropout
print(f"==> Loaded Model Weight from: {model_weight_path}")


# ==========================================
# 3. 测试函数定义：单张图片预测 + 测试集整体评估
# ==========================================
def predict_single_image(model, img_path, transform, classes, device):
    """单张图片预测：返回预测类别、概率、原图"""
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"加载图片失败：{e}，请检查图片路径是否正确")

    # 预处理+增加batch维度 [C,H,W] → [1,C,H,W]
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred_idx = torch.max(output, 1)
        pred_class = classes[pred_idx.item()]
        pred_prob = torch.softmax(output, dim=1)[0][pred_idx.item()].item()

    return pred_class, pred_prob, img


def evaluate_test_set(model, testloader, device, classes):
    """测试集整体评估：返回整体准确率、各类别准确率、各类别正确/总数"""
    total_correct = 0
    total_samples = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, pred_idx = torch.max(outputs, 1)

            # 统计整体
            total_samples += targets.size(0)
            total_correct += (pred_idx == targets).sum().item()

            # 统计各类别
            c = (pred_idx == targets).squeeze()
            for i in range(targets.size(0)):
                # 处理batch_size=1时的维度问题
                if targets.size(0) == 1:
                    label = targets[0]
                    class_correct[label] += c.item()
                    class_total[label] += 1
                else:
                    label = targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    # 计算准确率
    overall_acc = 100. * total_correct / total_samples
    class_acc = {classes[i]: 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in
                 range(len(classes))}

    # 把class_correct/class_total也返回，解决作用域问题
    return overall_acc, class_acc, class_correct, class_total


# ==========================================
# 4. 主测试逻辑：单张图片预测 / 测试集整体评估
# ==========================================
if __name__ == '__main__':
    # 模式1：单张图片预测（指定--img-path）
    if args.img_path is not None:
        if not os.path.exists(args.img_path):
            raise FileNotFoundError(f"图片文件不存在：{args.img_path}")
        pred_class, pred_prob, img = predict_single_image(net, args.img_path, transform, cifar10_classes, device)
        # 打印结果
        print("=" * 50)
        print(f"单张图片预测结果：")
        print(f"图片路径：{args.img_path}")
        print(f"预测类别：{pred_class}")
        print(f"预测概率：{pred_prob:.4f} ({pred_prob * 100:.2f}%)")
        print("=" * 50)
        # 显示图片
        img.show(title=f"Pred: {pred_class} ({pred_prob * 100:.2f}%)")

    # 模式2：测试集整体评估（默认）
    else:
        print("==> Preparing CIFAR-10 Test Set...")
        # 构建测试集DataLoader（复用顶部transform，和train.py一致）
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True
        )
        print(f"==> Test Set Size: {len(testset)} samples")

        # 执行评估（接收返回的class_correct/class_total，解决作用域问题）
        overall_acc, class_acc, class_correct, class_total = evaluate_test_set(net, testloader, device, cifar10_classes)

        # 打印详细结果（修复变量未定义问题）
        print("=" * 60)
        print(f"测试集整体准确率: {overall_acc:.2f}%")
        print("=" * 60)
        print("各类别准确率（正确数/总数）：")
        for i, cls in enumerate(cifar10_classes):
            print(f"  {cls:6s}: {class_acc[cls]:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        print("=" * 60)

        # 保存测试日志到./results
        log_path = os.path.join(args.save_dir, f'{model_name}_test_log.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Test Log for {model_name} | Device: {device}\n")
            f.write(f"Test Set Size: {len(testset)} samples\n")
            f.write(f"Overall Accuracy: {overall_acc:.2f}%\n")
            f.write("-" * 30 + "\n")
            f.write("Class-wise Accuracy (Correct/Total):\n")
            for i, cls in enumerate(cifar10_classes):
                f.write(f"  {cls:6s}: {class_acc[cls]:.2f}% ({int(class_correct[i])}/{int(class_total[i])})\n")
        print(f"==> Test Log Saved to: {log_path}")