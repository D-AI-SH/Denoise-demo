import math
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --- Dataset Configuration ---
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
NUM_IMAGES = 500
MIN_SHAPES_PER_IMAGE = 5
MAX_SHAPES_PER_IMAGE = 15
MASK_RATIO_RANGE = (0.01, 0.06)
MASK_BLOCK_SIZE_RANGE = (2, 6)
MASK_GRID_SIZE = 8
GAUSSIAN_NOISE_STD_RANGE = (0.12, 0.15)
BACKGROUND_COLOR_RANGE = (12, 96)
BACKGROUND_GRADIENT_STRENGTH = 0.10
BACKGROUND_TEXTURE_STD = 0.015


def get_random_shape_params(max_width, max_height):
    size = random.randint(20, 50)
    position = (
        random.randint(0, max_width - size),
        random.randint(0, max_height - size),
    )
    rotation = random.randint(0, 360)
    color = tuple(random.randint(40, 255) for _ in range(3))
    is_sharp = random.choice([True, False])
    return size, position, rotation, color, is_sharp


def paste_layer(canvas, layer, position, rotation, color, is_sharp):
    if not is_sharp:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 2.5)))
    rotated_layer = layer.rotate(rotation, expand=True, resample=Image.BICUBIC)
    mask = rotated_layer
    solid_color_layer = Image.new("RGB", rotated_layer.size, color)
    paste_x = position[0] - (rotated_layer.width - layer.width // 2) + layer.width // 4
    paste_y = position[1] - (rotated_layer.height - layer.height // 2) + layer.height // 4
    canvas.paste(solid_color_layer, (paste_x, paste_y), mask)


def draw_shape_on_layer(draw_func, size):
    layer_size = int(size * 1.5)
    layer = Image.new("L", (layer_size, layer_size), 0)
    draw = ImageDraw.Draw(layer)
    draw_func(draw, layer_size, size)
    return layer


def add_square(canvas, width, height):
    size, pos, rot, color, sharp = get_random_shape_params(width, height)

    def draw_func(draw, layer_size, shape_size):
        shape_pos = (layer_size - shape_size) // 2
        draw.rectangle(
            [shape_pos, shape_pos, shape_pos + shape_size, shape_pos + shape_size],
            fill=255,
        )

    layer = draw_shape_on_layer(draw_func, size)
    paste_layer(canvas, layer, pos, rot, color, sharp)


def add_circle(canvas, width, height):
    size, pos, rot, color, sharp = get_random_shape_params(width, height)

    def draw_func(draw, layer_size, shape_size):
        shape_pos = (layer_size - shape_size) // 2
        draw.ellipse(
            [shape_pos, shape_pos, shape_pos + shape_size, shape_pos + shape_size],
            fill=255,
        )

    layer = draw_shape_on_layer(draw_func, size)
    paste_layer(canvas, layer, pos, rot, color, sharp)


def add_triangle(canvas, width, height):
    size, pos, rot, color, sharp = get_random_shape_params(width, height)

    def draw_func(draw, layer_size, shape_size):
        center = layer_size // 2
        tri_height = shape_size * math.sqrt(3) / 2
        p1 = (center, center - tri_height / 2)
        p2 = (center - shape_size / 2, center + tri_height / 2)
        p3 = (center + shape_size / 2, center + tri_height / 2)
        draw.polygon([p1, p2, p3], fill=255)

    layer = draw_shape_on_layer(draw_func, size)
    paste_layer(canvas, layer, pos, rot, color, sharp)


def add_digit(canvas, width, height):
    size, pos, rot, color, sharp = get_random_shape_params(width, height)
    digit = str(random.randint(0, 9))
    try:
        font = ImageFont.truetype("arial.ttf", size=size)
    except IOError:
        font = ImageFont.load_default()

    def draw_func(draw, layer_size, shape_size):
        text_pos = (layer_size - shape_size) // 2
        draw.text((text_pos, text_pos), digit, font=font, fill=255)

    layer = draw_shape_on_layer(draw_func, size)
    paste_layer(canvas, layer, pos, rot, color, sharp)


def create_background_canvas(width, height):
    base_color = np.array(
        [random.randint(*BACKGROUND_COLOR_RANGE) for _ in range(3)],
        dtype=np.float32,
    ) / 255.0
    x_coords = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y_coords = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)

    background = np.zeros((height, width, 3), dtype=np.float32)
    for channel in range(3):
        x_slope = random.uniform(-BACKGROUND_GRADIENT_STRENGTH, BACKGROUND_GRADIENT_STRENGTH)
        y_slope = random.uniform(-BACKGROUND_GRADIENT_STRENGTH, BACKGROUND_GRADIENT_STRENGTH)
        background[..., channel] = base_color[channel] + x_slope * xx + y_slope * yy

    texture = np.random.normal(
        0.0, BACKGROUND_TEXTURE_STD, size=(height, width, 3)
    ).astype(np.float32)
    background = np.clip(background + texture, 0.0, 1.0)
    return Image.fromarray((background * 255.0).astype(np.uint8), mode="RGB")


def generate_clean_image(width, height, min_shapes, max_shapes):
    canvas = create_background_canvas(width, height)
    shape_functions = [add_circle, add_square, add_triangle, add_digit]
    num_shapes_to_draw = random.randint(min_shapes, max_shapes)

    for _ in range(num_shapes_to_draw):
        random.choice(shape_functions)(canvas, width, height)

    return np.array(canvas, dtype=np.float32) / 255.0


def apply_gaussian_noise(clean_image):
    noise_std = random.uniform(*GAUSSIAN_NOISE_STD_RANGE)
    noise = np.random.normal(0.0, noise_std, size=clean_image.shape).astype(np.float32)
    noisy_image = np.clip(clean_image + noise, 0.0, 1.0)
    return noisy_image, noise_std


def apply_random_mask(image):
    masked_image = image.copy()
    mask = np.ones(image.shape[:2], dtype=np.float32)
    target_ratio = random.uniform(*MASK_RATIO_RANGE)
    total_pixels = image.shape[0] * image.shape[1]
    masked_pixels = 0

    grid_h = max(1, image.shape[0] // MASK_GRID_SIZE)
    grid_w = max(1, image.shape[1] // MASK_GRID_SIZE)
    grid_cells = [(row, col) for row in range(MASK_GRID_SIZE) for col in range(MASK_GRID_SIZE)]
    random.shuffle(grid_cells)
    cell_index = 0

    while masked_pixels / total_pixels < target_ratio:
        if cell_index >= len(grid_cells):
            random.shuffle(grid_cells)
            cell_index = 0

        cell_row, cell_col = grid_cells[cell_index]
        cell_index += 1

        row_start = cell_row * grid_h
        col_start = cell_col * grid_w
        row_end = image.shape[0] if cell_row == MASK_GRID_SIZE - 1 else min(image.shape[0], row_start + grid_h)
        col_end = image.shape[1] if cell_col == MASK_GRID_SIZE - 1 else min(image.shape[1], col_start + grid_w)

        max_block_h = min(MASK_BLOCK_SIZE_RANGE[1], row_end - row_start)
        max_block_w = min(MASK_BLOCK_SIZE_RANGE[1], col_end - col_start)
        min_block_h = min(MASK_BLOCK_SIZE_RANGE[0], max_block_h)
        min_block_w = min(MASK_BLOCK_SIZE_RANGE[0], max_block_w)

        if max_block_h <= 0 or max_block_w <= 0:
            continue

        block_h = random.randint(min_block_h, max_block_h)
        block_w = random.randint(min_block_w, max_block_w)
        top = random.randint(row_start, row_end - block_h)
        left = random.randint(col_start, col_end - block_w)

        mask[top : top + block_h, left : left + block_w] = 0.0
        masked_image[top : top + block_h, left : left + block_w, :] = 0.0
        masked_pixels = int((mask == 0).sum())

    return masked_image, mask


def create_degraded_sample(clean_image):
    noisy_image, noise_std = apply_gaussian_noise(clean_image)
    degraded_image, mask = apply_random_mask(noisy_image)
    return degraded_image, mask, noise_std


def save_visualization(clean_image, degraded_image, mask, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(clean_image)
    axes[0].set_title("Clean RGB")
    axes[0].axis("off")

    axes[1].imshow(degraded_image)
    axes[1].set_title("Noisy + Masked")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Valid-Pixel Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    data_dir = "data"

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    print(f"--- Generating RGB denoising dataset in '{data_dir}' ---")

    for i in range(NUM_IMAGES):
        sample_dir = os.path.join(data_dir, f"{i:03d}")
        os.makedirs(sample_dir)

        clean_image = generate_clean_image(
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            MIN_SHAPES_PER_IMAGE,
            MAX_SHAPES_PER_IMAGE,
        )
        degraded_image, mask, noise_std = create_degraded_sample(clean_image)

        np.save(os.path.join(sample_dir, "clean.npy"), clean_image.astype(np.float32))
        np.save(os.path.join(sample_dir, "degraded.npy"), degraded_image.astype(np.float32))
        np.save(os.path.join(sample_dir, "mask.npy"), mask.astype(np.float32))

        save_visualization(
            clean_image,
            degraded_image,
            mask,
            os.path.join(sample_dir, "comparison.png"),
        )

        with open(os.path.join(sample_dir, "meta.txt"), "w", encoding="utf-8") as meta_file:
            meta_file.write(f"gaussian_noise_std={noise_std:.6f}\n")
            meta_file.write(f"masked_ratio={(mask == 0).mean():.6f}\n")

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{NUM_IMAGES} samples...")

    print("--- Dataset generation finished successfully ---")
