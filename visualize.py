import os
# 导入 os 模块，用于检查目录是否存在、拼接保存路径

# 导入 matplotlib 的 pyplot 接口，并简写成 plt，用来画图
import matplotlib.pyplot as plt
# 导入 numpy 并简写成 np，用于数组运算
import numpy as np
# 导入 PyTorch，用于张量处理
import torch


# 定义函数 build_model_input：把退化图和掩码拼接成模型的输入
def build_model_input(degraded_input, mask):
    # torch.cat 沿 dim=1（通道维）拼接两个张量
    return torch.cat([degraded_input, mask], dim=1)


# 定义内部函数 _setup_fonts：设置 matplotlib 的中英文字体
def _setup_fonts():
    # try 用来尝试执行可能出错的代码
    try:
        # 设置字体族为无衬线字体
        plt.rcParams["font.family"] = "sans-serif"
        # 设置可用的中英文字体列表
        plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun"]
        # 让坐标轴上的负号正常显示，而不是方块
        plt.rcParams["axes.unicode_minus"] = False
    # 如果上面任何一行出错，就捕获异常并保存到 exc
    except Exception as exc:
        # 用 f-string 打印提示信息，告诉用户字体设置失败
        print(f"Font setup failed, fallback to default fonts: {exc}")


# 定义函数 _to_hwc_image：把 CHW 张量转成适合 matplotlib 显示的 HWC numpy 数组
def _to_hwc_image(tensor):
    # detach() 断开梯度；cpu() 移到内存；clamp(0,1) 限制像素范围；permute(1,2,0) 把 CHW 改成 HWC；numpy() 转成数组
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    # 返回处理后的数组
    return array


# 定义函数 plot_loss_curve：画训练和验证损失曲线；output_dir 默认是 "visualization"
def plot_loss_curve(train_losses, val_losses, output_dir="visualization"):
    # 先设置字体
    _setup_fonts()
    # 创建一个字典 title_font，用来指定标题字体属性：加粗、字号 12
    title_font = {"weight": "bold", "size": 12}

    # 创建一张 10x6 英寸的新图
    plt.figure(figsize=(10, 6))
    # range(1, len+1) 生成 1,2,3,... 作为横坐标（epoch 编号）
    epochs = range(1, len(train_losses) + 1)
    # 画训练损失曲线：蓝色、圆点、实线，图例名称是 "Training Loss"
    plt.plot(epochs, train_losses, "b-o", label="Training Loss")
    # 画验证损失曲线：红色、圆点、实线
    plt.plot(epochs, val_losses, "r-o", label="Validation Loss")
    # 设置图的标题，fontdict 传入字体设置
    plt.title("Training and Validation Loss Curve", fontdict=title_font)
    # 设置横轴标签
    plt.xlabel("Epochs")
    # 设置纵轴标签
    plt.ylabel("Loss (Log Scale)")
    # 把纵轴改成对数刻度，方便观察损失下降
    plt.yscale("log")
    # 显示图例
    plt.legend()
    # 画网格线；which="both" 表示主刻度和次刻度都画，ls="--" 表示虚线
    plt.grid(True, which="both", ls="--")

    # 如果输出目录不存在
    if not os.path.exists(output_dir):
        # 就递归创建这个目录
        os.makedirs(output_dir)

    # 拼接出图片保存路径
    save_path = os.path.join(output_dir, "loss_curve.png")
    # 把当前图保存到 save_path
    plt.savefig(save_path)
    # 关闭当前图，释放内存
    plt.close()
    # 打印保存位置
    print(f"Loss curve saved to '{save_path}'")


# 定义函数 visualize_results：在数据集上取若干样本，生成去噪效果对比图
def visualize_results(model, dataset, device, num_samples=5, output_dir="visualization"):
    # 设置字体
    _setup_fonts()
    # 设置标题字体
    title_font = {"weight": "bold", "size": 12}
    # 模型切到评估模式，避免训练层影响结果
    model.eval()

    # 如果输出目录不存在
    if not os.path.exists(output_dir):
        # 就创建它
        os.makedirs(output_dir)

    # min 取较小值，防止要生成的样本数超过数据集长度
    num_samples = min(num_samples, len(dataset))
    # 打印将要生成的样本数
    print(f"Generating {num_samples} denoising visualizations...")

    # 循环生成 num_samples 张对比图
    for i in range(num_samples):
        # 从数据集按下标取一个样本：退化图、干净图、掩码
        degraded_input, clean_target, mask = dataset[i]
        # unsqueeze(0) 在开头加一维变成批次 [1, C, H, W]，再移到指定设备
        degraded_batch = degraded_input.unsqueeze(0).to(device)
        # 掩码同样加批次维并移动设备
        mask_batch = mask.unsqueeze(0).to(device)

        # 推理阶段不需要计算梯度，所以用 no_grad 包住
        with torch.no_grad():
            # 把退化图和掩码拼接后送进模型
            denoised_output = model(
                # 调用拼接函数生成模型输入
                build_model_input(degraded_batch, mask_batch)
            # 模型输出是批次形式，squeeze(0) 去掉批次维
            ).squeeze(0)

        # 把退化图张量转成 HWC numpy 数组
        degraded_np = _to_hwc_image(degraded_input)
        # 把模型输出去噪图转成 HWC numpy 数组
        denoised_np = _to_hwc_image(denoised_output)
        # 把干净图转成 HWC numpy 数组
        clean_np = _to_hwc_image(clean_target)
        # np.abs 取绝对值，再对 RGB 三个通道求平均，得到每个像素的误差图
        diff_map = np.mean(np.abs(denoised_np - clean_np), axis=2)
        # 把掩码去掉批次维并转成 numpy，用于和误差图相乘
        mask_np = mask.squeeze(0).cpu().numpy()

        # 创建一行 4 个子图的画布，总宽度 16 英寸，每个子图高 4 英寸
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        # 在第一个子图上显示退化图
        axes[0].imshow(degraded_np)
        # 给第一个子图加标题
        axes[0].set_title("Degraded Input", fontdict=title_font)
        # 关闭第一个子图的坐标轴
        axes[0].axis("off")

        # 第二个子图显示模型输出去噪图
        axes[1].imshow(denoised_np)
        # 加标题
        axes[1].set_title("Model Output", fontdict=title_font)
        # 关闭坐标轴
        axes[1].axis("off")

        # 第三个子图显示干净原图
        axes[2].imshow(clean_np)
        # 加标题
        axes[2].set_title("Ground Truth", fontdict=title_font)
        # 关闭坐标轴
        axes[2].axis("off")

        # 第四个子图显示掩码后的误差；乘以 mask_np 后只在有效像素上显示误差
        im = axes[3].imshow(diff_map * mask_np, cmap="magma")
        # 给误差子图加标题
        axes[3].set_title("Masked Abs Error", fontdict=title_font)
        # 关闭坐标轴
        axes[3].axis("off")
        # 在误差子图旁边添加颜色条，显示数值和颜色的对应关系
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        # 自动调整子图间距，避免重叠
        plt.tight_layout()
        # 保存当前对比图，文件名是 comparison_编号.png
        plt.savefig(os.path.join(output_dir, f"comparison_{i}.png"))
        # 关闭当前图释放内存
        plt.close(fig)

    # 打印结果保存目录
    print(f"Visualizations saved to '{output_dir}' directory.")
