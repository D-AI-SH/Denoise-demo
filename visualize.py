import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def build_model_input(degraded_input, mask):
    return torch.cat([degraded_input, mask], dim=1)


def _setup_fonts():
    try:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as exc:
        print(f"Font setup failed, fallback to default fonts: {exc}")


def _to_hwc_image(tensor):
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return array


def plot_loss_curve(train_losses, val_losses, output_dir="visualization"):
    _setup_fonts()
    title_font = {"weight": "bold", "size": 12}

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-o", label="Training Loss")
    plt.plot(epochs, val_losses, "r-o", label="Validation Loss")
    plt.title("Training and Validation Loss Curve", fontdict=title_font)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to '{save_path}'")


def visualize_results(model, dataset, device, num_samples=5, output_dir="visualization"):
    _setup_fonts()
    title_font = {"weight": "bold", "size": 12}
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_samples = min(num_samples, len(dataset))
    print(f"Generating {num_samples} denoising visualizations...")

    for i in range(num_samples):
        degraded_input, clean_target, mask = dataset[i]
        degraded_batch = degraded_input.unsqueeze(0).to(device)
        mask_batch = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            denoised_output = model(
                build_model_input(degraded_batch, mask_batch)
            ).squeeze(0)

        degraded_np = _to_hwc_image(degraded_input)
        denoised_np = _to_hwc_image(denoised_output)
        clean_np = _to_hwc_image(clean_target)
        diff_map = np.mean(np.abs(denoised_np - clean_np), axis=2)
        mask_np = mask.squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(degraded_np)
        axes[0].set_title("Degraded Input", fontdict=title_font)
        axes[0].axis("off")

        axes[1].imshow(denoised_np)
        axes[1].set_title("Model Output", fontdict=title_font)
        axes[1].axis("off")

        axes[2].imshow(clean_np)
        axes[2].set_title("Ground Truth", fontdict=title_font)
        axes[2].axis("off")

        im = axes[3].imshow(diff_map * mask_np, cmap="magma")
        axes[3].set_title("Masked Abs Error", fontdict=title_font)
        axes[3].axis("off")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{i}.png"))
        plt.close(fig)

    print(f"Visualizations saved to '{output_dir}' directory.")
