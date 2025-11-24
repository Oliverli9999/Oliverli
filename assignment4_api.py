# assignment4_api.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
from fastapi import FastAPI
from typing import Dict

# ---------- Device ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------- CIFAR-10 class names ----------
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ---------- Transforms (must match training) ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# ============= 1. Energy Model (same as训练时) =============
class EnergyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32 -> 16
        x = self.pool(F.relu(self.conv2(x)))   # 16 -> 8
        x = self.pool(F.relu(self.conv3(x)))   # 8 -> 4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

    def energy(self, x):
        logits = self.forward(x)
        max_logit, _ = torch.max(logits, dim=1)
        return -max_logit  # negative confidence


# ============= 2. Diffusion UNet (简化版，和训练时一致) =============
class SimpleUNet3C(nn.Module):
    def __init__(self, img_channels=3, base_channels=64, diffusion_steps=200):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        in_channels = img_channels + 1  # extra time channel

        self.down1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

        self.act = nn.ReLU()

    def forward(self, x, t):
        # time embedding as extra channel in [0,1]
        t = t.float() / (self.diffusion_steps - 1)
        t = t.view(-1, 1, 1, 1)
        t = t.expand(-1, 1, x.size(2), x.size(3))
        x = torch.cat([x, t], dim=1)

        d1 = self.act(self.down1(x))
        d2 = self.act(self.down2(self.pool(d1)))
        u1 = self.act(self.up1(d2))
        out = self.out_conv(u1)
        return out  # predicted noise


# ---------- Diffusion schedule helpers ----------
def get_diffusion_params(T, device):
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


@torch.no_grad()
def sample_and_save(model, betas, alphas, alpha_bar,
                    num_samples=16, diffusion_steps=200,
                    filename="diffusion_samples_api.png"):
    model.to(device)
    model.eval()

    C, H, W = 3, 32, 32
    x_t = torch.randn(num_samples, C, H, W, device=device)

    for t in reversed(range(diffusion_steps)):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        eps_theta = model(x_t, t_tensor)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (x_t - coef2 * eps_theta)

        if t > 0:
            z = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_t = mean + sigma_t * z
        else:
            x_t = mean

    samples = (x_t.clamp(-1, 1) + 1) / 2.0
    vutils.save_image(samples, filename, nrow=4, padding=2)
    return filename


# ============= 3. Load trained models =============
energy_model = EnergyCNN(num_classes=10).to(device)
energy_model.load_state_dict(torch.load("energy_cnn.pth", map_location=device))
energy_model.eval()

diffusion_steps = 200  # must match training
diffusion_model = SimpleUNet3C(img_channels=3,
                               base_channels=64,
                               diffusion_steps=diffusion_steps).to(device)
diffusion_model.load_state_dict(torch.load("diffusion_unet_cifar10.pth", map_location=device))
diffusion_model.eval()

betas, alphas, alpha_bar = get_diffusion_params(diffusion_steps, device)

# ============= 4. FastAPI app =============
app = FastAPI(title="Assignment4 Energy & Diffusion API")


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Assignment4 Energy & Diffusion API is running"}


@app.get("/energy_example")
def energy_example() -> Dict[str, float | str]:
    """
    Take one CIFAR-10 test image, run through the energy model,
    and return the true label, predicted label, and energy value.
    """
    testset = datasets.CIFAR10(root="./data",
                               train=False,
                               download=True,
                               transform=transform)
    img, label = testset[0]  # just use the first test image
    true_label = CIFAR10_CLASSES[label]

    with torch.no_grad():
        img_batch = img.unsqueeze(0).to(device)
        logits = energy_model(img_batch)
        energy = energy_model.energy(img_batch)[0].item()
        pred_idx = torch.argmax(logits, dim=1)[0].item()
        pred_label = CIFAR10_CLASSES[pred_idx]

    return {
        "true_label": true_label,
        "predicted_label": pred_label,
        "energy": energy,
    }


@app.get("/diffusion_generate")
def diffusion_generate() -> Dict[str, str]:
    """
    Generate 16 CIFAR-like samples from the diffusion model,
    save them to a PNG file, and return the filename.
    """
    filename = "diffusion_samples_api.png"
    saved_file = sample_and_save(
        diffusion_model,
        betas,
        alphas,
        alpha_bar,
        num_samples=16,
        diffusion_steps=diffusion_steps,
        filename=filename,
    )
    return {
        "message": "Diffusion samples generated and saved.",
        "file": saved_file,
    }
