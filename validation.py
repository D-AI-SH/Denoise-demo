import torch
# 导入 PyTorch 主模块，用于张量运算
from tqdm import tqdm
# 从 tqdm 库导入 tqdm：它能把循环包装成带进度条的迭代器


# 定义函数 build_model_input，两个参数是退化图和掩码
def build_model_input(degraded_input, mask):
    # torch.cat 在指定维度上拼接张量；dim=1 表示沿通道维拼接，得到 [B, 4, H, W] 输入
    return torch.cat([degraded_input, mask], dim=1)


# 定义验证函数：model 是网络，dataloader 是数据加载器，criterion 是损失函数，device 是 CPU/GPU 设备
def validate(model, dataloader, criterion, device):
    # 切换到评估模式；这会关闭 Dropout 等只在训练时生效的层
    model.eval()
    # 用一个浮点数累加器记录所有样本的损失总和
    total_loss = 0.0

    # 把数据加载器包进 tqdm，显示验证进度条；leave=False 表示结束后不保留进度条
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    # with torch.no_grad() 块内不计算梯度，节省显存并加速推理
    with torch.no_grad():
        # 遍历数据加载器；每次解包出退化图、干净目标图和掩码三个张量
        for degraded_input, clean_target, mask in progress_bar:
            # 把退化图移动到指定设备（GPU 或 CPU）
            degraded_input = degraded_input.to(device)
            # 把干净目标图移动到指定设备
            clean_target = clean_target.to(device)
            # 把掩码移动到指定设备
            mask = mask.to(device)

            # 把退化图和掩码拼接后送入模型，得到去噪输出
            denoised_output = model(build_model_input(degraded_input, mask))
            # 用损失函数计算模型输出和干净图之间的误差
            loss = criterion(denoised_output, clean_target)
            # 用 .item() 把标量张量取出为普通数字，并累加进总损失
            total_loss += loss.item()
            # 在进度条右侧显示当前批次的损失
            progress_bar.set_postfix({"loss": loss.item()})

    # 返回平均损失：总损失除以数据加载器中的批次数量
    return total_loss / len(dataloader)
