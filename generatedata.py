import math
# 导入 math 模块：提供平方根等数学函数
import os
# 导入 os 模块：负责路径拼接、目录判断和创建
import random
# 导入 random 模块：负责生成随机数
import shutil
# 导入 shutil 模块：用于递归删除整个目录

# 导入 matplotlib 的 pyplot 接口，简写成 plt，用于生成样本预览图
import matplotlib.pyplot as plt
# 导入 numpy 并简写成 np，用于生成背景、添加噪声等数组运算
import numpy as np
# 从 PIL 导入图像类 Image、绘图类 ImageDraw、滤镜类 ImageFilter、字体类 ImageFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# 下面这一段都是数据集配置常量
# --- Dataset Configuration ---
# 常量：生成图像的宽度
IMAGE_WIDTH = 128
# 常量：生成图像的高度
IMAGE_HEIGHT = 128
# 常量：一共生成多少张图像
NUM_IMAGES = 500
# 常量：每张图至少画的形状数量
MIN_SHAPES_PER_IMAGE = 5
# 常量：每张图最多画的形状数量
MAX_SHAPES_PER_IMAGE = 15
# 常量：掩码覆盖像素比例的范围（1% 到 6%）
MASK_RATIO_RANGE = (0.01, 0.06)
# 常量：单个掩码块大小的范围（2 到 6 像素）
MASK_BLOCK_SIZE_RANGE = (2, 6)
# 常量：掩码网格是 8x8
MASK_GRID_SIZE = 8
# 常量：高斯噪声标准差的范围
GAUSSIAN_NOISE_STD_RANGE = (0.12, 0.15)
# 常量：背景基础颜色的取值范围（RGB 0-255）
BACKGROUND_COLOR_RANGE = (12, 96)
# 常量：背景渐变强度
BACKGROUND_GRADIENT_STRENGTH = 0.10
# 常量：背景纹理噪声标准差
BACKGROUND_TEXTURE_STD = 0.015


# 定义函数 get_random_shape_params：随机生成一个形状的尺寸、位置、旋转、颜色和锐利程度
def get_random_shape_params(max_width, max_height):
    # random.randint(20, 50) 在 20 到 50 之间随机选一个整数作为形状大小
    size = random.randint(20, 50)
    # 生成一个元组，包含随机 x 和 y 坐标，确保形状不会超出图片范围
    position = (
        # x 坐标从 0 到 max_width - size 之间随机选
        random.randint(0, max_width - size),
        # y 坐标从 0 到 max_height - size 之间随机选
        random.randint(0, max_height - size),
    # 结束 position 元组
    )
    # 随机旋转角度 0 到 360 度
    rotation = random.randint(0, 360)
    # 生成一个包含三个随机整数的元组，作为 RGB 颜色；生成器表达式会执行三次
    color = tuple(random.randint(40, 255) for _ in range(3))
    # random.choice 从 [True, False] 中随机选一个，决定形状是否模糊
    is_sharp = random.choice([True, False])
    # 返回这五个随机参数
    return size, position, rotation, color, is_sharp


# 定义函数 paste_layer：把一个形状图层旋转后粘贴到画布上
def paste_layer(canvas, layer, position, rotation, color, is_sharp):
    # 如果形状不是锐利的
    if not is_sharp:
        # 对图层做高斯模糊，模糊半径随机取 1 到 2.5
        layer = layer.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 2.5)))
    # 用 rotate 旋转图层；expand=True 扩大画布容纳旋转后的内容，BICUBIC 是高质量插值
    rotated_layer = layer.rotate(rotation, expand=True, resample=Image.BICUBIC)
    # 把旋转后的灰度图层当作透明掩码，后面用 mask 决定哪些像素被粘贴
    mask = rotated_layer
    # 创建一个和旋转图层同样大小的纯色 RGB 图像，颜色是随机生成的 color
    solid_color_layer = Image.new("RGB", rotated_layer.size, color)
    # 计算粘贴的 x 坐标：根据旋转后宽度和原图层宽度做近似居中
    paste_x = position[0] - (rotated_layer.width - layer.width // 2) + layer.width // 4
    # 计算粘贴的 y 坐标
    paste_y = position[1] - (rotated_layer.height - layer.height // 2) + layer.height // 4
    # 把纯色图层按 mask 粘贴到画布上，mask 中白色区域会被粘贴
    canvas.paste(solid_color_layer, (paste_x, paste_y), mask)


# 定义函数 draw_shape_on_layer：在独立的灰度图层上画出指定形状
def draw_shape_on_layer(draw_func, size):
    # 图层边长设为形状大小的 1.5 倍，给旋转留空间
    layer_size = int(size * 1.5)
    # 创建一张 "L" 模式的灰度图像，初始全部是 0（黑色）
    layer = Image.new("L", (layer_size, layer_size), 0)
    # 创建画笔对象，用于在 layer 上绘图
    draw = ImageDraw.Draw(layer)
    # 调用传入的 draw_func 函数，把具体形状画到图层上
    draw_func(draw, layer_size, size)
    # 返回画好的图层
    return layer


# 定义函数 add_square：在画布上添加一个正方形
def add_square(canvas, width, height):
    # 调用随机参数函数，获取正方形的大小、位置、旋转、颜色和锐利程度
    size, pos, rot, color, sharp = get_random_shape_params(width, height)

    # 定义一个内部函数 draw_func：负责在指定大小的图层里画正方形
    def draw_func(draw, layer_size, shape_size):
        # 计算正方形左上角坐标，使其在图层中居中
        shape_pos = (layer_size - shape_size) // 2
        # draw.rectangle 画矩形；列表给出左上角和右下角坐标，fill=255 表示白色
        draw.rectangle(
            # 矩形的四个顶点坐标
            [shape_pos, shape_pos, shape_pos + shape_size, shape_pos + shape_size],
            # 填充色为 255（白色）
            fill=255,
        # 结束 draw.rectangle 调用
        )

    # 先画到独立图层上
    layer = draw_shape_on_layer(draw_func, size)
    # 再把图层旋转、上色并粘贴到画布
    paste_layer(canvas, layer, pos, rot, color, sharp)


# 定义函数 add_circle：在画布上添加一个圆形
def add_circle(canvas, width, height):
    # 获取圆形随机参数
    size, pos, rot, color, sharp = get_random_shape_params(width, height)

    # 内部函数：在图层上画圆形
    def draw_func(draw, layer_size, shape_size):
        # 计算圆心/外接正方形左上角，使其居中
        shape_pos = (layer_size - shape_size) // 2
        # draw.ellipse 画椭圆，这里边长相等所以是正圆
        draw.ellipse(
            # 外接矩形的两个对角坐标
            [shape_pos, shape_pos, shape_pos + shape_size, shape_pos + shape_size],
            # 填充白色
            fill=255,
        # 结束 draw.ellipse 调用
        )

    # 画到图层上
    layer = draw_shape_on_layer(draw_func, size)
    # 粘贴到画布
    paste_layer(canvas, layer, pos, rot, color, sharp)


# 定义函数 add_triangle：在画布上添加一个三角形
def add_triangle(canvas, width, height):
    # 获取三角形随机参数
    size, pos, rot, color, sharp = get_random_shape_params(width, height)

    # 内部函数：在图层上画三角形
    def draw_func(draw, layer_size, shape_size):
        # 计算图层中心点
        center = layer_size // 2
        # 计算等边三角形的高度：边长乘以 sqrt(3) 再除以 2
        tri_height = shape_size * math.sqrt(3) / 2
        # 顶点坐标：位于中心上方半个三角形高度处
        p1 = (center, center - tri_height / 2)
        # 左下角坐标
        p2 = (center - shape_size / 2, center + tri_height / 2)
        # 右下角坐标
        p3 = (center + shape_size / 2, center + tri_height / 2)
        # 用三个顶点画多边形，填充白色
        draw.polygon([p1, p2, p3], fill=255)

    # 画到图层上
    layer = draw_shape_on_layer(draw_func, size)
    # 粘贴到画布
    paste_layer(canvas, layer, pos, rot, color, sharp)


# 定义函数 add_digit：在画布上添加一个 0-9 的数字
def add_digit(canvas, width, height):
    # 获取数字随机参数
    size, pos, rot, color, sharp = get_random_shape_params(width, height)
    # 随机选一个 0 到 9 的数字，转成字符串
    digit = str(random.randint(0, 9))
    # 尝试加载 arial 字体，字号等于形状大小
    try:
        # ImageFont.truetype 从系统字体文件加载 TrueType 字体
        font = ImageFont.truetype("arial.ttf", size=size)
    # 如果找不到 arial.ttf，会抛出 IOError
    except IOError:
        # 使用 PIL 自带的默认字体作为替代
        font = ImageFont.load_default()

    # 内部函数：在图层上写字
    def draw_func(draw, layer_size, shape_size):
        # 计算文字左上角坐标，使其大致居中
        text_pos = (layer_size - shape_size) // 2
        # draw.text 在指定位置画文字，白色填充
        draw.text((text_pos, text_pos), digit, font=font, fill=255)

    # 画到图层上
    layer = draw_shape_on_layer(draw_func, size)
    # 粘贴到画布
    paste_layer(canvas, layer, pos, rot, color, sharp)


# 定义函数 create_background_canvas：生成一张带渐变和纹理的随机背景图
def create_background_canvas(width, height):
    # np.array 创建 RGB 三个通道的基础颜色数组
    base_color = np.array(
        # 对三个通道分别从颜色范围中随机取一个整数
        [random.randint(*BACKGROUND_COLOR_RANGE) for _ in range(3)],
        # 指定数组元素类型为 float32
        dtype=np.float32,
    # 结束 np.array 的调用
    ) / 255.0
    # 除以 255.0 把颜色从 0-255 归一化到 0-1 范围
    x_coords = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    # 生成横坐标序列，范围从 -1 到 1，共 width 个点
    y_coords = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    # 生成纵坐标序列
    xx, yy = np.meshgrid(x_coords, y_coords)
    # np.meshgrid 生成二维网格坐标，xx 和 yy 的尺寸匹配图片大小

    # 创建一个 height x width x 3 的零数组，用于存放背景
    background = np.zeros((height, width, 3), dtype=np.float32)
    # 对 RGB 三个通道逐个处理
    for channel in range(3):
        # 随机生成这个通道在 x 方向上的渐变斜率
        x_slope = random.uniform(-BACKGROUND_GRADIENT_STRENGTH, BACKGROUND_GRADIENT_STRENGTH)
        # 随机生成这个通道在 y 方向上的渐变斜率
        y_slope = random.uniform(-BACKGROUND_GRADIENT_STRENGTH, BACKGROUND_GRADIENT_STRENGTH)
        # 基础颜色加上 x 和 y 方向的线性渐变
        background[..., channel] = base_color[channel] + x_slope * xx + y_slope * yy

    # 生成高斯噪声数组，形状和背景相同，模拟纹理
    texture = np.random.normal(
        # 均值 0，标准差 BACKGROUND_TEXTURE_STD，输出尺寸为图片大小
        0.0, BACKGROUND_TEXTURE_STD, size=(height, width, 3)
    # 结束 np.random.normal 调用
    ).astype(np.float32)
    # 把背景和纹理相加，并用 np.clip 限制在 0 到 1 之间
    background = np.clip(background + texture, 0.0, 1.0)
    # 把浮点数组乘 255 转成 0-255 的 uint8，再创建成 PIL RGB 图像
    return Image.fromarray((background * 255.0).astype(np.uint8), mode="RGB")


# 定义函数 generate_clean_image：生成一张没有噪声和掩码的干净图片
def generate_clean_image(width, height, min_shapes, max_shapes):
    # 先创建背景画布
    canvas = create_background_canvas(width, height)
    # 列出可用的四种形状函数
    shape_functions = [add_circle, add_square, add_triangle, add_digit]
    # 随机决定这张图要画多少个形状
    num_shapes_to_draw = random.randint(min_shapes, max_shapes)

    # 循环画 num_shapes_to_draw 个形状
    for _ in range(num_shapes_to_draw):
        # 随机选一个形状函数，并传入画布和图片尺寸执行
        random.choice(shape_functions)(canvas, width, height)

    # 把 PIL 画布转成 numpy 数组，除以 255 归一化到 0-1，并转成 float32
    return np.array(canvas, dtype=np.float32) / 255.0


# 定义函数 apply_gaussian_noise：给干净图添加高斯噪声
def apply_gaussian_noise(clean_image):
    # 从噪声范围中随机取一个标准差；* 号表示把元组展开成两个参数
    noise_std = random.uniform(*GAUSSIAN_NOISE_STD_RANGE)
    # 生成和图像形状相同的高斯噪声
    noise = np.random.normal(0.0, noise_std, size=clean_image.shape).astype(np.float32)
    # 噪声叠加到干净图上，并裁剪到 0-1
    noisy_image = np.clip(clean_image + noise, 0.0, 1.0)
    # 返回带噪声的图像和本次噪声标准差
    return noisy_image, noise_std


# 定义函数 apply_random_mask：在图像上随机遮挡若干小块像素
def apply_random_mask(image):
    # copy() 复制一份图像，避免修改原数组
    masked_image = image.copy()
    # 创建全 1 的掩码，1 表示有效像素，0 表示被遮挡
    mask = np.ones(image.shape[:2], dtype=np.float32)
    # 从掩码比例范围中随机取一个目标遮挡比例
    target_ratio = random.uniform(*MASK_RATIO_RANGE)
    # 计算图像总像素数
    total_pixels = image.shape[0] * image.shape[1]
    # 记录当前已经被遮挡的像素数量
    masked_pixels = 0

    # 计算每个网格单元的高度，至少为 1
    grid_h = max(1, image.shape[0] // MASK_GRID_SIZE)
    # 计算每个网格单元的宽度，至少为 1
    grid_w = max(1, image.shape[1] // MASK_GRID_SIZE)
    # 列表推导式生成所有网格单元坐标，共 8x8 个
    grid_cells = [(row, col) for row in range(MASK_GRID_SIZE) for col in range(MASK_GRID_SIZE)]
    # 随机打乱网格顺序
    random.shuffle(grid_cells)
    # 用下标记录当前取到第几个网格
    cell_index = 0

    # 当遮挡比例还没达到目标时，继续遮挡
    while masked_pixels / total_pixels < target_ratio:
        # 如果所有网格都用完了
        if cell_index >= len(grid_cells):
            # 重新打乱网格顺序
            random.shuffle(grid_cells)
            # 下标归零，从头再来
            cell_index = 0

        # 取出当前网格的行和列
        cell_row, cell_col = grid_cells[cell_index]
        # 下标加一，指向下一个网格
        cell_index += 1

        # 计算这个网格的起始行
        row_start = cell_row * grid_h
        # 计算这个网格的起始列
        col_start = cell_col * grid_w
        # 计算结束行；最后一个网格直接到图像底部，否则限制在网格范围
        row_end = image.shape[0] if cell_row == MASK_GRID_SIZE - 1 else min(image.shape[0], row_start + grid_h)
        # 计算结束列
        col_end = image.shape[1] if cell_col == MASK_GRID_SIZE - 1 else min(image.shape[1], col_start + grid_w)

        # 最大块高度不能超过网格高度和配置上限
        max_block_h = min(MASK_BLOCK_SIZE_RANGE[1], row_end - row_start)
        # 最大块宽度不能超过网格宽度和配置上限
        max_block_w = min(MASK_BLOCK_SIZE_RANGE[1], col_end - col_start)
        # 最小块高度，同时不能超过最大块高度
        min_block_h = min(MASK_BLOCK_SIZE_RANGE[0], max_block_h)
        # 最小块宽度
        min_block_w = min(MASK_BLOCK_SIZE_RANGE[0], max_block_w)

        # 如果这个网格放不下一块像素，就跳过
        if max_block_h <= 0 or max_block_w <= 0:
            # 跳过当前网格，继续下一个
            continue

        # 随机决定这块遮挡的高度
        block_h = random.randint(min_block_h, max_block_h)
        # 随机决定这块遮挡的宽度
        block_w = random.randint(min_block_w, max_block_w)
        # 随机决定块的顶部位置
        top = random.randint(row_start, row_end - block_h)
        # 随机决定块的左侧位置
        left = random.randint(col_start, col_end - block_w)

        # 把掩码中这个矩形区域设为 0，表示像素无效
        mask[top : top + block_h, left : left + block_w] = 0.0
        # 把图像中这个矩形区域设为 0，表示像素缺失
        masked_image[top : top + block_h, left : left + block_w, :] = 0.0
        # 重新统计掩码中等于 0 的像素数量，并转成整数
        masked_pixels = int((mask == 0).sum())

    # 返回被遮挡的图像和掩码
    return masked_image, mask


# 定义函数 create_degraded_sample：把干净图变成一张带噪声和掩码的退化图
def create_degraded_sample(clean_image):
    # 先添加高斯噪声
    noisy_image, noise_std = apply_gaussian_noise(clean_image)
    # 再在噪声图上添加随机遮挡
    degraded_image, mask = apply_random_mask(noisy_image)
    # 返回退化图、掩码和噪声标准差
    return degraded_image, mask, noise_std


# 定义函数 save_visualization：把干净图、退化图、掩码保存成一张预览图
def save_visualization(clean_image, degraded_image, mask, output_path):
    # 创建一行 3 个子图的画布
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 第一个子图显示干净图
    axes[0].imshow(clean_image)
    # 设置标题
    axes[0].set_title("Clean RGB")
    # 关闭坐标轴
    axes[0].axis("off")

    # 第二个子图显示带噪声和遮挡的图
    axes[1].imshow(degraded_image)
    # 设置标题
    axes[1].set_title("Noisy + Masked")
    # 关闭坐标轴
    axes[1].axis("off")

    # 第三个子图用灰度显示掩码
    axes[2].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
    # 设置标题
    axes[2].set_title("Valid-Pixel Mask")
    # 关闭坐标轴
    axes[2].axis("off")

    # 自动调整子图布局
    plt.tight_layout()
    # 保存图片到指定路径
    plt.savefig(output_path)
    # 关闭当前图释放内存
    plt.close(fig)


# 这是入口判断：直接运行本文件时才执行下面的数据生成流程
if __name__ == "__main__":
    # 设置数据保存目录为当前目录下的 data 文件夹
    data_dir = "data"

    # 如果目录已经存在
    if os.path.exists(data_dir):
        # 递归删除旧目录，避免新旧数据混在一起
        shutil.rmtree(data_dir)
    # 创建新的数据目录
    os.makedirs(data_dir)
    # 打印开始生成提示
    print(f"--- Generating RGB denoising dataset in '{data_dir}' ---")

    # 循环生成 NUM_IMAGES（500）张样本
    for i in range(NUM_IMAGES):
        # 为每个样本创建独立子目录，名称是三位编号
        sample_dir = os.path.join(data_dir, f"{i:03d}")
        # 创建样本目录
        os.makedirs(sample_dir)

        # 生成干净图
        clean_image = generate_clean_image(
            # 传入图像宽度
            IMAGE_WIDTH,
            # 传入图像高度
            IMAGE_HEIGHT,
            # 传入最少形状数
            MIN_SHAPES_PER_IMAGE,
            # 传入最多形状数
            MAX_SHAPES_PER_IMAGE,
        # 结束 generate_clean_image 调用
        )
        # 生成退化图、掩码和噪声标准差
        degraded_image, mask, noise_std = create_degraded_sample(clean_image)

        # 把干净图保存成 clean.npy
        np.save(os.path.join(sample_dir, "clean.npy"), clean_image.astype(np.float32))
        # 把退化图保存成 degraded.npy
        np.save(os.path.join(sample_dir, "degraded.npy"), degraded_image.astype(np.float32))
        # 把掩码保存成 mask.npy
        np.save(os.path.join(sample_dir, "mask.npy"), mask.astype(np.float32))

        # 生成一张预览对比图
        save_visualization(
            # 传入干净图
            clean_image,
            # 传入退化图
            degraded_image,
            # 传入掩码
            mask,
            # 指定预览图的保存路径
            os.path.join(sample_dir, "comparison.png"),
        # 结束 save_visualization 调用
        )

        # 打开（或创建）meta.txt 文本文件；with 会在代码块结束后自动关闭文件
        with open(os.path.join(sample_dir, "meta.txt"), "w", encoding="utf-8") as meta_file:
            # 向文件写入一行噪声标准差信息
            meta_file.write(f"gaussian_noise_std={noise_std:.6f}\n")
            # 向文件写入一行遮挡比例信息
            meta_file.write(f"masked_ratio={(mask == 0).mean():.6f}\n")

        # 每处理完 20 张就打印一次进度
        if (i + 1) % 20 == 0:
            # 打印当前进度
            print(f"  Processed {i + 1}/{NUM_IMAGES} samples...")

    # 全部生成完成后打印成功提示
    print("--- Dataset generation finished successfully ---")
