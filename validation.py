import torch
from tqdm import tqdm


def build_model_input(degraded_input, mask):
    return torch.cat([degraded_input, mask], dim=1)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for degraded_input, clean_target, mask in progress_bar:
            degraded_input = degraded_input.to(device)
            clean_target = clean_target.to(device)
            mask = mask.to(device)

            denoised_output = model(build_model_input(degraded_input, mask))
            loss = criterion(denoised_output, clean_target)
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)
