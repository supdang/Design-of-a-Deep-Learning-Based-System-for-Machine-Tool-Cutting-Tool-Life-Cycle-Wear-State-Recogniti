# 断点保存和恢复功能使用指南

本项目已添加断点保存和恢复功能，支持CPU和GPU训练。该功能允许您：

1. 在训练过程中保存当前状态（断点）
2. 从断点处恢复训练，无需从头开始
3. 支持CPU和GPU训练

## 新增功能文件

- `train_with_checkpoint.py` - 通用训练脚本（支持CPU和GPU，支持断点）
- `train_gpu.py` - 专门的GPU训练脚本（支持断点）
- `train_cpu.py` - CPU训练脚本（已更新，支持断点）

## 使用方法

### 1. 通用训练脚本（推荐）

```bash
# CPU训练
python train_with_checkpoint.py --device cpu --epochs 100 --batch-size 32

# GPU训练
python train_with_checkpoint.py --device cuda --epochs 100 --batch-size 32

# 自动选择设备（如果有GPU则使用GPU，否则使用CPU）
python train_with_checkpoint.py --device auto --epochs 100 --batch-size 32

# 从断点恢复训练
python train_with_checkpoint.py --resume --checkpoint-path checkpoints/my_checkpoint.pth

# 指定自定义断点路径
python train_with_checkpoint.py --checkpoint-path checkpoints/custom_checkpoint.pth
```

### 2. 专门的GPU训练脚本

```bash
# 正常GPU训练
python train_gpu.py --epochs 100 --batch-size 32

# 从断点恢复GPU训练
python train_gpu.py --resume --checkpoint-path checkpoints/gpu_checkpoint.pth

# 快速GPU训练（测试用）
python train_gpu.py --quick
```

### 3. CPU训练脚本（已更新）

```bash
# CPU训练（现在也支持断点）
python train_cpu.py

# 快速CPU训练（测试用）
python train_cpu.py --quick
```

## 断点功能说明

### 断点保存内容

断点文件包含以下信息：

- 模型权重和结构
- 优化器状态
- 学习率调度器状态
- 训练历史（损失和准确率）
- 最佳模型状态
- 当前训练轮数
- 最佳验证准确率
- 早停相关计数器

### 断点文件位置

默认断点文件保存在以下位置：

- CPU训练: `checkpoints/cpu_training_checkpoint.pth`
- GPU训练: `checkpoints/gpu_training_checkpoint.pth`
- 通用训练: `checkpoints/training_checkpoint.pth`
- 快速训练: 相应的快速训练断点文件

## API接口

在代码中，您可以使用以下方法：

```python
from src.models.lstm_model import ToolWearClassifier

classifier = ToolWearClassifier(device='cuda')

# 训练时启用断点功能
history = classifier.train(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    checkpoint_path='my_checkpoint.pth',      # 断点文件路径
    resume_from_checkpoint=True,              # 是否从断点恢复
    # ... 其他参数
)

# 手动保存断点
classifier.save_checkpoint(
    file_path='manual_checkpoint.pth',
    optimizer_state=classifier.optimizer.state_dict(),
    epoch=50
)

# 手动加载断点
epoch, best_epoch, no_improve_epochs = classifier.load_checkpoint('manual_checkpoint.pth')
```

## 注意事项

1. 断点功能会增加磁盘空间使用，因为需要保存完整的训练状态
2. 从断点恢复时，模型结构应与保存时保持一致
3. GPU训练的断点文件可能在CPU上恢复，反之亦然（PyTorch会自动处理设备映射）
4. 断点文件通常比最终模型文件大，因为包含更多训练状态信息