import numpy as np
# 导入 numpy 并简写成 np，用于随机数、数组和索引操作
import torch
# 导入 PyTorch 主模块
import torch.nn as nn
# 导入 PyTorch 的神经网络模块，简写成 nn，用于定义损失函数等
import torch.optim as optim
# 导入 PyTorch 的优化器模块，简写成 optim，用于创建 Adam 优化器
from torch.optim.lr_scheduler import StepLR
# 从 PyTorch 导入 StepLR 学习率调度器，用于每隔若干轮降低学习率
from torch.utils.data import DataLoader, Subset
# 从 PyTorch 导入 DataLoader（批量加载数据）和 Subset（取数据集子集）
from tqdm import tqdm
# 从 tqdm 导入进度条工具

# 从 model.py 导入自定义网络 RGBDenoiseNet
from model import RGBDenoiseNet
# 从 utils.py 导入自定义数据集 DenoisingDataset
from utils import DenoisingDataset
# 从 validation.py 导入验证函数 validate
from validation import validate
# 从 visualize.py 导入画损失曲线和可视化结果的函数
from visualize import plot_loss_curve, visualize_results

# 常量：数据集根目录（训练时请改成你自己的数据路径）
DATA_DIR = "h:/chaofen/data"
# 常量：每个批次放 25 张图
BATCH_SIZE = 25
# 常量：初始学习率 0.0006
LEARNING_RATE = 0.0006
# 常量：总共训练 300 轮
EPOCHS = 300
# 常量：模型权重保存的文件名
MODEL_SAVE_PATH = "rgb_denoise_net.pth"
# 常量：每 30 轮调整一次学习率
SCHEDULER_STEP_SIZE = 30
# 常量：每次调整学习率乘以 0.5
SCHEDULER_GAMMA = 0.5
# 常量：随机种子，保证结果可复现
RANDOM_SEED = 42


# 定义函数 build_model_input：拼接退化图和掩码
def build_model_input(degraded_input, mask):
    # torch.cat 沿通道维 dim=1 拼接，得到 [B, 4, H, W] 的模型输入
    return torch.cat([degraded_input, mask], dim=1)


# 定义训练函数：model 网络、dataloader 数据、optimizer 优化器、criterion 损失函数、device 设备
def train(model, dataloader, optimizer, criterion, device):
    # 切换到训练模式，BatchNorm 和 Dropout 会按训练行为运行
    model.train()
    # 累加器变量，记录一个 epoch 内所有批次的损失之和
    total_loss = 0.0

    # 用 tqdm 包装数据加载器，显示训练进度条
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    # 遍历数据加载器，每次解包出退化图、干净图、掩码
    for degraded_input, clean_target, mask in progress_bar:
        # 把退化图移动到指定设备（GPU 或 CPU）
        degraded_input = degraded_input.to(device)
        # 把干净图移动到指定设备
        clean_target = clean_target.to(device)
        # 把掩码移动到指定设备
        mask = mask.to(device)

        # 清空上一步累积的梯度，否则梯度会叠加
        optimizer.zero_grad()
        # 拼接输入后送入模型，得到去噪输出
        restored_output = model(build_model_input(degraded_input, mask))
        # 计算模型输出和干净图之间的损失
        loss = criterion(restored_output, clean_target)
        # 反向传播：根据损失自动计算每个参数的梯度
        loss.backward()
        # 优化器根据梯度更新模型参数
        optimizer.step()

        # 用 .item() 把损失张量变成普通数字并累加
        total_loss += loss.item()
        # 在进度条右侧显示当前批次损失
        progress_bar.set_postfix({"loss": loss.item()})

    # 返回平均损失：总损失除以批次数量
    return total_loss / len(dataloader)


# 这是 Python 的入口判断：只有直接运行 run.py 时才执行下面代码
if __name__ == "__main__":
    # 设置 PyTorch 随机种子，使模型初始化可复现
    torch.manual_seed(RANDOM_SEED)
    # 设置 numpy 随机种子，使数据划分可复现
    np.random.seed(RANDOM_SEED)

    # 创建设备对象：有 CUDA 就用 GPU，否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # f-string 把设备名称打印出来
    print(f"Using device: {device}")

    # 读取完整数据集
    full_dataset = DenoisingDataset(data_dir=DATA_DIR)
    # 获取数据集样本总数
    dataset_size = len(full_dataset)
    # np.arange 生成 0 到 dataset_size-1 的整数数组，作为样本索引
    indices = np.arange(dataset_size)
    # 随机打乱索引顺序
    np.random.shuffle(indices)

    # 训练集数量：70% 的样本，int() 取整
    train_size = int(0.7 * dataset_size)
    # 验证集数量：10% 的样本
    val_size = int(0.1 * dataset_size)

    # 切片取出前 train_size 个索引作为训练集索引
    train_indices = indices[:train_size]
    # 切片取出接下来 val_size 个索引作为验证集索引
    val_indices = indices[train_size : train_size + val_size]
    # 剩下的索引全部作为测试集
    test_indices = indices[train_size + val_size :]

    # Subset 把完整数据集包装成只含指定索引的子集；tolist() 把 numpy 数组转成列表
    train_dataset = Subset(full_dataset, train_indices.tolist())
    # 创建验证子集
    val_dataset = Subset(full_dataset, val_indices.tolist())
    # 创建测试子集
    test_dataset = Subset(full_dataset, test_indices.tolist())

    # DataLoader 负责自动按批次、随机顺序加载训练数据；num_workers=0 表示主进程加载
    train_loader = DataLoader(
        # 传入训练子集
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    # 结束训练加载器的创建
    )
    # 创建验证加载器；shuffle=False 表示不打乱顺序
    val_loader = DataLoader(
        # 传入验证子集
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    # 结束验证加载器的创建
    )
    # 创建测试加载器
    test_loader = DataLoader(
        # 传入测试子集
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    # 结束测试加载器的创建
    )

    # 打印数据划分信息；print 参数跨多行书写，功能不变
    print(
        # f-string 可以插入变量值；字符串太长时拆成两段相邻字符串会自动拼接
        f"Dataset split: {len(train_dataset)} training, "
        # 第二段字符串继续补充验证集和测试集数量
        f"{len(val_dataset)} validation, {len(test_dataset)} test samples."
    # 结束 print 调用
    )

    # 创建去噪网络，输入 4 通道、输出 3 通道，并移动到指定设备
    model = RGBDenoiseNet(in_channels=4, out_channels=3).to(device)
    # 使用均方误差损失（MSE），衡量像素值差异
    criterion = nn.MSELoss()
    # 使用 Adam 优化器，传入模型所有参数和学习率
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 创建 StepLR 学习率调度器
    scheduler = StepLR(
        # 调度器作用在 optimizer 上，每隔 STEP_SIZE 轮把学习率乘以 gamma
        optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
    # 结束调度器创建
    )

    # 打印训练开始提示
    print("Starting RGB denoising training...")
    # 创建空列表，用来记录每一轮训练损失
    train_losses = []
    # 创建空列表，用来记录每一轮验证损失
    val_losses = []
    # for 循环跑 EPOCHS 轮训练；range(EPOCHS) 生成 0,1,...,299
    for epoch in range(EPOCHS):
        # 调用训练函数，得到这一轮的平均训练损失
        avg_train_loss = train(model, train_loader, optimizer, criterion, device)
        # 调用验证函数，得到这一轮的平均验证损失
        avg_val_loss = validate(model, val_loader, criterion, device)
        # 更新学习率：到指定步数后自动降低
        scheduler.step()

        # 把训练损失追加进列表
        train_losses.append(avg_train_loss)
        # 把验证损失追加进列表
        val_losses.append(avg_val_loss)
        # 打印本轮信息
        print(
            # 显示当前是第几轮；epoch + 1 是因为 Python 从 0 开始计数
            f"Epoch [{epoch + 1}/{EPOCHS}], "
            # 显示训练损失，保留 6 位小数
            f"Train Loss: {avg_train_loss:.6f}, "
            # 显示验证损失
            f"Val Loss: {avg_val_loss:.6f}, "
            # 显示当前学习率；get_last_lr() 返回列表，取第 0 个
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        # 结束 print
        )

    # torch.save 保存模型权重到指定文件
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    # 打印保存完成信息
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")

    # 打印测试集评估提示
    print("\nRunning final evaluation on the test set...")
    # 在测试集上计算最终损失
    test_loss = validate(model, test_loader, criterion, device)
    # 打印测试损失
    print(f"Final Test Loss: {test_loss:.6f}")

    # 画出训练和验证损失曲线，保存到 test_visualizations 目录
    plot_loss_curve(train_losses, val_losses, output_dir="test_visualizations")

    # 打印开始生成可视化提示
    print("\nGenerating visualizations on test data...")
    # 调用可视化函数
    visualize_results(
        # 传入训练好的模型
        model,
        # 传入测试数据集
        test_dataset,
        # 传入计算设备
        device,
        # 指定生成 5 张对比图
        num_samples=5,
        # 指定输出目录
        output_dir="test_visualizations",
    # 结束 visualize_results 调用
    )
