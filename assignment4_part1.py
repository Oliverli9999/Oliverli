# assignment4_part1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# 0. Device & Dataloaders
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def get_cifar10_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),                      # [0,1]
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),     # [-1,1]
    ])
    train_set = datasets.CIFAR10(root="./data",
                                 train=True,
                                 download=True,
                                 transform=transform)
    test_set = datasets.CIFAR10(root="./data",
                                train=False,
                                download=True,
                                transform=transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2)
    return train_loader, test_loader


# =========================
# 1. Energy Model (simple CNN classifier)
# =========================
class EnergyCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification.
    You can interpret the negative class logit as "energy".
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # (B,32,32,32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # (B,64,16,16)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # (B,128,8,8)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32 -> 16
        x = self.pool(F.relu(self.conv2(x)))   # 16 -> 8
        x = self.pool(F.relu(self.conv3(x)))   # 8  -> 4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

    def energy(self, x):
        """
        Example: use negative max logit as an "energy".
        Lower energy = more confident.
        """
        logits = self.forward(x)
        max_logit, _ = torch.max(logits, dim=1)
        return -max_logit


def train_energy_model(epochs=5, lr=1e-3, batch_size=64):
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    model = EnergyCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Energy Epoch {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        print(f"[Energy] Epoch {epoch} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    # Simple test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"[Energy] Test Accuracy: {correct / total:.4f}")

    # Save model
    torch.save(model.state_dict(), "energy_cnn.pth")
    print("Saved energy model to energy_cnn.pth")
    return model


# =========================
# 2. Diffusion Model
# =========================
def get_diffusion_params(T, device):
    """
    Simple linear beta schedule.
    """
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def q_sample(x0, t, alpha_bar, noise=None):
    """
    Forward diffusion:
      x_t = sqrt(alpha_bar_t)*x0 + sqrt(1 - alpha_bar_t)*noise
    """
    if noise is None:
        noise = torch.randn_like(x0)
    a_t = alpha_bar[t].view(-1, 1, 1, 1)  # (B,) -> (B,1,1,1)
    x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1.0 - a_t) * noise
    return x_t, noise


class SimpleUNet3C(nn.Module):
    """
    Very small UNet-like model for CIFAR-10 (3x32x32) with time conditioning.
    """

    def __init__(self, img_channels=3, base_channels=64, diffusion_steps=1000):
        super().__init__()
        self.diffusion_steps = diffusion_steps

        in_channels = img_channels + 1  # extra channel for time

        self.down1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

        self.act = nn.ReLU()

    def forward(self, x, t):
        """
        x: (B,3,32,32)
        t: (B,)
        """
        # time embedding as an extra channel
        t = t.float() / (self.diffusion_steps - 1)
        t = t.view(-1, 1, 1, 1)
        t = t.expand(-1, 1, x.size(2), x.size(3))   # (B,1,H,W)

        x = torch.cat([x, t], dim=1)  # (B,4,H,W)

        d1 = self.act(self.down1(x))            # (B,64,H,W)
        d2 = self.act(self.down2(self.pool(d1)))  # (B,128,H/2,W/2)

        u1 = self.act(self.up1(d2))             # (B,64,H,W)
        out = self.out_conv(u1)                 # (B,3,H,W)

        return out  # predicted noise


def train_diffusion_model(epochs=1, lr=1e-4, batch_size=64, diffusion_steps=1000):
    train_loader, _ = get_cifar10_loaders(batch_size=batch_size)
    model = SimpleUNet3C(img_channels=3,
                         base_channels=64,
                         diffusion_steps=diffusion_steps).to(device)
    betas, alphas, alpha_bar = get_diffusion_params(diffusion_steps, device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for images, _ in tqdm(train_loader, desc=f"Diffusion Epoch {epoch}/{epochs}"):
            images = images.to(device)

            B = images.size(0)
            t = torch.randint(0, diffusion_steps, (B,), device=device)

            x_t, noise = q_sample(images, t, alpha_bar)
            pred_noise = model(x_t, t)

            loss = criterion(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Diffusion] Epoch {epoch} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "diffusion_unet_cifar10.pth")
    print("Saved diffusion model to diffusion_unet_cifar10.pth")
    return model, (betas, alphas, alpha_bar)


@torch.no_grad()
def sample_from_diffusion(model, betas, alphas, alpha_bar,
                          num_samples=16, diffusion_steps=1000):
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

    # map back from [-1,1] to [0,1] for visualization
    samples = (x_t.clamp(-1, 1) + 1) / 2.0
    grid = vutils.make_grid(samples, nrow=4, padding=2)

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title("Diffusion CIFAR-10 Samples")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()


# =========================
# 3. Main
# =========================
if __name__ == "__main__":
    # 1) Train Energy Model
    energy_model = train_energy_model(epochs=3, lr=1e-3, batch_size=64)

    # 2) Train Diffusion Model (short training just to demonstrate)
    diffusion_model, diff_params = train_diffusion_model(
        epochs=1, lr=1e-4, batch_size=64, diffusion_steps=200
    )

    # 3) Sample images from Diffusion Model
    betas, alphas, alpha_bar = diff_params
    sample_from_diffusion(diffusion_model, betas, alphas, alpha_bar,
                          num_samples=16, diffusion_steps=200)
