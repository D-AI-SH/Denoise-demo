import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import RGBDenoiseNet
from utils import DenoisingDataset
from validation import validate
from visualize import plot_loss_curve, visualize_results

DATA_DIR = "h:/chaofen/data"
BATCH_SIZE = 25
LEARNING_RATE = 0.0006
EPOCHS = 300
MODEL_SAVE_PATH = "rgb_denoise_net.pth"
SCHEDULER_STEP_SIZE = 30
SCHEDULER_GAMMA = 0.5
RANDOM_SEED = 42


def build_model_input(degraded_input, mask):
    return torch.cat([degraded_input, mask], dim=1)


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for degraded_input, clean_target, mask in progress_bar:
        degraded_input = degraded_input.to(device)
        clean_target = clean_target.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        restored_output = model(build_model_input(degraded_input, mask))
        loss = criterion(restored_output, clean_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = DenoisingDataset(data_dir=DATA_DIR)
    dataset_size = len(full_dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    train_size = int(0.7 * dataset_size)
    val_size = int(0.1 * dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_dataset = Subset(full_dataset, train_indices.tolist())
    val_dataset = Subset(full_dataset, val_indices.tolist())
    test_dataset = Subset(full_dataset, test_indices.tolist())

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(
        f"Dataset split: {len(train_dataset)} training, "
        f"{len(val_dataset)} validation, {len(test_dataset)} test samples."
    )

    model = RGBDenoiseNet(in_channels=4, out_channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(
        optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
    )

    print("Starting RGB denoising training...")
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        avg_train_loss = train(model, train_loader, optimizer, criterion, device)
        avg_val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], "
            f"Train Loss: {avg_train_loss:.6f}, "
            f"Val Loss: {avg_val_loss:.6f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")

    print("\nRunning final evaluation on the test set...")
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.6f}")

    plot_loss_curve(train_losses, val_losses, output_dir="test_visualizations")

    print("\nGenerating visualizations on test data...")
    visualize_results(
        model,
        test_dataset,
        device,
        num_samples=5,
        output_dir="test_visualizations",
    )
