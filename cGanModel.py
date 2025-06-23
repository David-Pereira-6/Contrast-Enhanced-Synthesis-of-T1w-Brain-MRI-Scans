import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import mean_squared_error as mse_metric

# -----------------------------
# 1. Parâmetros (otimizados para desempenho)
# -----------------------------
batch_size = 2  # Reduzido para poupar memória
lr = 0.0002
epochs = 45
img_size = 128

# -----------------------------
# 2. Dataset
# -----------------------------
class PairedMRIDataset(Dataset):
    def __init__(self, t1_dir, t1gd_dir, slice_range=(59, 100)):
        self.t1_dir = t1_dir
        self.t1gd_dir = t1gd_dir
        self.t1_files = sorted(os.listdir(t1_dir))
        self.t1gd_files = sorted(os.listdir(t1gd_dir))
        self.slice_range = slice_range
        self.data = self._build_index()

    def _build_index(self):
        index = []
        for i in range(len(self.t1_files)):
            t1_path = os.path.join(self.t1_dir, self.t1_files[i])
            try:
                t1_img = nib.load(t1_path).get_fdata()
                max_slices = t1_img.shape[2]
                for z in range(self.slice_range[0], min(self.slice_range[1], max_slices)):
                    index.append((i, z))
            except Exception as e:
                print(f"Erro ao processar {t1_path}: {e}")
        return index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.data[idx]
        t1_path = os.path.join(self.t1_dir, self.t1_files[vol_idx])
        t1gd_path = os.path.join(self.t1gd_dir, self.t1gd_files[vol_idx])

        t1_img = nib.load(t1_path).get_fdata()
        t1gd_img = nib.load(t1gd_path).get_fdata()

        t1_slice = t1_img[:, :, slice_idx].astype(np.float32)
        t1gd_slice = t1gd_img[:, :, slice_idx].astype(np.float32)

        # Normalização baseada no volume completo (não slice-a-slice)
        t1_slice = (t1_slice - np.min(t1_img)) / (np.max(t1_img) - np.min(t1_img) + 1e-8)
        t1gd_slice = (t1gd_slice - np.min(t1gd_img)) / (np.max(t1gd_img) - np.min(t1gd_img) + 1e-8)

        t1_slice = torch.from_numpy(t1_slice).unsqueeze(0)
        t1gd_slice = torch.from_numpy(t1gd_slice).unsqueeze(0)

        t1_slice = nn.functional.interpolate(t1_slice.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)
        t1gd_slice = nn.functional.interpolate(t1gd_slice.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)

        t1_slice = t1_slice * 2 - 1
        t1gd_slice = t1gd_slice * 2 - 1

        return t1_slice, t1gd_slice

# -----------------------------
# 3. Generator U-Net com Dropout e skip connections
# -----------------------------
class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()

        def down_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.5)  # Temporariamente desativado para diagnóstico
            )

        self.down1 = down_block(1, 64)    # 120x120
        self.down2 = down_block(64, 128)  # 60x60
        self.down3 = down_block(128, 256) # 30x30
        self.down4 = down_block(256, 512) # 15x15

        self.up1 = up_block(512, 256)     # 30x30
        self.up2 = up_block(512, 128)     # 60x60
        self.up3 = up_block(256, 64)      # 120x120
        self.up4 = up_block(128, 1)       # 240x240

        self.final = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        u2 = self.up2(torch.cat((u1, d3), dim=1))
        u3 = self.up3(torch.cat((u2, d2), dim=1))
        u4 = self.up4(torch.cat((u3, d1), dim=1))
        return self.final(u4)

# -----------------------------
# 4. Discriminator (PatchGAN)
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input_img, generated_img):
        x = torch.cat((input_img, generated_img), 1)
        return self.model(x)

# -----------------------------
# 5. Inicialização
# -----------------------------
t1_dir = "C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/t1"
t1gd_dir = "C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/t1Gd"
dataset = PairedMRIDataset(t1_dir, t1gd_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

generator = GeneratorUNet()
discriminator = Discriminator()

criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()


optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# -----------------------------
# 6. Loop de treino Pix2Pix
# -----------------------------
epoch_losses_G = []
epoch_losses_D = []
epoch_mse = []
epoch_psnr = []
epoch_ssim = []

for epoch in range(epochs):
    epoch_loss_G = 0
    epoch_loss_D = 0

    for i, (t1, t1gd) in enumerate(dataloader):
        fake_t1gd = generator(t1)
        pred_fake = discriminator(t1, fake_t1gd)
        valid = torch.ones_like(pred_fake)
        fake = torch.zeros_like(pred_fake)

        optimizer_G.zero_grad()
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_L1 = criterion_L1(fake_t1gd, t1gd)
        loss_G = loss_GAN + 100 * loss_L1  # SSIM removido da loss
        loss_G.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        pred_real = discriminator(t1, t1gd)
        loss_real = criterion_GAN(pred_real, valid)
        pred_fake = discriminator(t1, fake_t1gd.detach())
        loss_fake = criterion_GAN(pred_fake, fake)
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()

        print(f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(dataloader)}] Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    epoch_losses_G.append(epoch_loss_G / len(dataloader))
    epoch_losses_D.append(epoch_loss_D / len(dataloader))

    # -----------------------------
    # Save checkpoint image por epoch e calcular métricas
    # -----------------------------
    generator.eval()
    with torch.no_grad():
        metrics_mse, metrics_psnr, metrics_ssim = [], [], []

        for k in range(3):  # Avaliar sobre 3 imagens distintas
            t1_example, t1gd_example = dataset[k]
            t1_example = t1_example.unsqueeze(0)
            fake = generator(t1_example)

            fake_img = ((fake[0].squeeze().cpu().numpy() + 1) / 2)
            real_img = ((t1gd_example.squeeze().cpu().numpy() + 1) / 2)
            input_img = ((t1_example.squeeze().cpu().numpy() + 1) / 2)

            mse_val = mse_metric(real_img, fake_img)
            psnr_val = psnr_metric(real_img, fake_img, data_range=1.0)
            ssim_val = ssim(fake, t1gd_example.unsqueeze(0), data_range=2.0).item()
            metrics_mse.append(mse_val)
            metrics_psnr.append(psnr_val)
            metrics_ssim.append(ssim_val)

        epoch_mse.append(np.mean(metrics_mse))
        epoch_psnr.append(np.mean(metrics_psnr))
        epoch_ssim.append(np.mean(metrics_ssim))

        print(f"Epoch {epoch+1} — MSE: {epoch_mse[-1]:.4f}, PSNR: {epoch_psnr[-1]:.2f}dB, SSIM: {epoch_ssim[-1]:.4f}")

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(input_img, cmap="gray", vmin=0, vmax=1)
        axs[0].set_title("Entrada T1")
        axs[1].imshow(real_img, cmap="gray", vmin=0, vmax=1)
        axs[1].set_title("Real T1Gd")
        axs[2].imshow(fake_img, cmap="gray", vmin=0, vmax=1)
        axs[2].set_title(f"Gerado T1Gd - Epoch {epoch+1}")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"output_epoch_{epoch+1}.png")
        plt.close()

        torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")

    generator.train()

# -----------------------------
# 7. Guardar modelo final e gráficos
# -----------------------------
os.makedirs("C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/modelo_final", exist_ok=True)
torch.save(generator.state_dict(), "C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/modelo_final/generator.pth")
torch.save(discriminator.state_dict(), "C:/Users/David Pereira/Desktop/David/uni/Mestrado/1º ano - 2º semestre/DAC/TrabalhoFinal/modelo_final/discriminator.pth")

# Gráficos
plt.figure()
plt.plot(epoch_losses_G, label='Loss Generator')
plt.plot(epoch_losses_D, label='Loss Discriminator')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('Evolution of Losses')
plt.savefig("losses_plot.png")
plt.close()

plt.figure()
plt.plot(epoch_mse, label='MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE por Época')
plt.grid(True)
plt.savefig("mse_plot.png")
plt.close()

plt.figure()
plt.plot(epoch_psnr, label='PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.title('PSNR por Época')
plt.grid(True)
plt.savefig("psnr_plot.png")
plt.close()

plt.figure()
plt.plot(epoch_ssim, label='SSIM')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.title('SSIM por Época')
plt.grid(True)
plt.savefig("ssim_plot.png")
plt.close()

# Média das últimas 10 épocas
if len(epoch_mse) >= 10:
    print("\nMédias das últimas 10 épocas:")
    print(f"MSE médio: {np.mean(epoch_mse[-10:]):.4f}")
    print(f"PSNR médio: {np.mean(epoch_psnr[-10:]):.2f} dB")
    print(f"SSIM médio: {np.mean(epoch_ssim[-10:]):.4f}")
