import matplotlib.pyplot as plt
import os


def plot_history(train_losses, test_accs, save_dir, model_name):
    """
    绘制训练过程中的 Loss 和 Accuracy 曲线并保存
    """
    epochs = range(1, len(train_losses) + 1)

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 绘制 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss.png'))
    plt.close()

    # 2. 绘制 Accuracy 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    plt.title(f'{model_name} - Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_accuracy.png'))
    plt.close()

    print(f"==> Plots saved to {save_dir}")


def count_parameters(model):
    """
    计算模型参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)