import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import math
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torchinfo import summary
from timm.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class EnhancedWaveletTransform2D(nn.Module):
    def __init__(self, channels, wavelet='db2', level=2):
        super().__init__()
        self.channels = channels
        self.wavelet = wavelet
        self.level = level
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.refine = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.InstanceNorm2d(channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_wave = []
        for b in range(B):
            for c in range(C):
                tensor = x[b, c].detach().cpu()
                coeffs = pywt.wavedec2(tensor.numpy(), self.wavelet, level=self.level)
                high_coeffs = coeffs[1:]
                rec = pywt.waverec2([coeffs[0]] + high_coeffs, self.wavelet)
                rec = torch.from_numpy(rec).to(x.device).unsqueeze(0).unsqueeze(0)
                rec = F.interpolate(rec, (H, W), mode='bilinear', align_corners=True)
                x_wave.append(rec)
        x_wave = torch.cat(x_wave, dim=0).view(B, C, H, W)
        x_wave = x_wave * self.attn(x_wave)
        x_wave = self.refine(x_wave)
        x_wave = self.norm(x_wave)
        x_wave = self.act(x_wave)
        return x_wave

class FourierLowPass2D(nn.Module):
    def __init__(self, channels, low_ratio=0.3):
        super().__init__()
        self.channels = channels
        self.low_ratio = low_ratio
        self.fuse = nn.Conv2d(channels, channels, 1, padding=0)
        self.norm = nn.InstanceNorm2d(channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shift = torch.fft.fftshift(x_fft)
        h_low = int(H * self.low_ratio // 2)
        w_low = int(W * self.low_ratio // 2)
        h_c, w_c = H//2, W//2
        mask = torch.zeros_like(x_fft_shift, dtype=torch.bool, device=x.device)
        mask[:, :, h_c-h_low:h_c+h_low, w_c-w_low:w_c+w_low] = True
        x_low_fft = x_fft_shift * mask
        x_low = torch.fft.ifft2(torch.fft.ifftshift(x_low_fft), dim=(-2, -1)).real
        x_low = self.fuse(x_low)
        x_low = self.norm(x_low)
        x_low = self.act(x_low)
        return x_low

class HypergraphConstructUnwrap(nn.Module):
    def __init__(self, in_channels, out_channels, k=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.construct = nn.Conv2d(
            in_channels*2, in_channels*2*k, 3, padding=1, groups=in_channels*2, bias=False
        )
        self.unwrap = nn.Sequential(
            nn.Conv2d(in_channels*2*k, out_channels, 1, padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.shortcut = nn.Conv2d(in_channels*2, out_channels, 1, padding=0) if in_channels*2 != out_channels else nn.Identity()

    def forward(self, x_high, x_low):
        x_cat = torch.cat([x_high, x_low], dim=1)
        x_hyper = self.construct(x_cat)
        x_unwrap = self.unwrap(x_hyper)
        x_out = x_unwrap + self.shortcut(x_cat)
        return x_out

class CascadeEncoderFirst(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.input_proj = nn.Conv2d(input_channels, out_channels, 3, padding=1)
        self.high_branch = EnhancedWaveletTransform2D(out_channels)
        self.low_branch = FourierLowPass2D(out_channels)
        self.hypergraph = HypergraphConstructUnwrap(out_channels, out_channels)
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        x_proj = self.input_proj(x)
        x_high = self.high_branch(x_proj)
        x_low = self.low_branch(x_proj)
        features = self.hypergraph(x_high, x_low)
        pooled = self.downsample(features)
        return features, pooled

class LRGS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k, padding=k//2, groups=in_channels),
                nn.InstanceNorm2d(in_channels),
                nn.LeakyReLU(inplace=True)
            ) for k in kernel_sizes
        ])
        self.fuse = nn.Conv2d(in_channels*len(kernel_sizes), out_channels, 1, padding=0)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        scale_feats = [conv(x) for conv in self.scale_convs]
        x_fuse = torch.cat(scale_feats, dim=1)
        x_fuse = self.fuse(x_fuse)
        x_fuse = self.norm(x_fuse)
        x_fuse = self.act(x_fuse)
        return x_fuse + self.shortcut(x)

class MC_Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ss2d = SS2D(d_model=d_model, d_state=d_state, dropout=dropout)
        self.channel_fuse = nn.Conv2d(d_model, d_model, 1, padding=0)
        self.norm = nn.InstanceNorm2d(d_model)

    def forward(self, x):
        x_mamba = x.permute(0, 2, 3, 1)
        x_mamba = self.ss2d(x_mamba)
        x_mamba = x_mamba.permute(0, 3, 1, 2)
        x_mamba = self.channel_fuse(x_mamba)
        x_mamba = self.norm(x_mamba)
        return x + x_mamba

class SS2D(nn.Module):
    def __init__(
            self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto",
            dt_min=0.001, dt_max=0.1, dropout=0., bias=False
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, d_conv, padding=d_conv//2, groups=self.d_inner)
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self._init_dt_proj()

        self.A_log = self._init_A_log(d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.selective_scan = selective_scan_fn

    def _init_dt_proj(self):
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt)

    def _init_A_log(self, d_state):
        A = torch.arange(1, d_state+1, dtype=torch.float32).repeat(self.d_inner, 1)
        A_log = torch.log(A)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    def forward(self, x):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0,3,1,2)
        x = self.act(self.conv2d(x))
        B, C, H, W = x.shape
        L = H * W
        x_flat = x.view(B, C, L).transpose(1,2)
        x_proj = self.x_proj(x_flat)
        dt, B_proj, C_proj = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).transpose(1,2)
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        out = self.selective_scan(
            x_flat.float().transpose(1,2), dt.float(),
            A, B_proj.transpose(1,2), C_proj.transpose(1,2), D,
            delta_bias=self.dt_proj.bias.float(), delta_softplus=True
        )
        out = out.transpose(1,2).view(B, H, W, C)
        out = self.out_norm(out)
        out = out * F.silu(z)
        out = self.out_proj(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class LRGS_MC_Mamba_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lrgs = LRGS(in_channels, out_channels)
        self.mc_mamba = MC_Mamba(out_channels)

    def forward(self, x):
        x = self.lrgs(x)
        x = self.mc_mamba(x)
        return x

class CascadeEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lrgs_mamba = LRGS_MC_Mamba_Block(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.lrgs_mamba(x)
        pooled = self.downsample(features)
        return features, pooled

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block1 = LRGS_MC_Mamba_Block(in_channels, out_channels)
        self.block2 = LRGS_MC_Mamba_Block(out_channels, out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lrgs_mamba = LRGS_MC_Mamba_Block(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        diffY = skip_connection.size(2) - x.size(2)
        diffX = skip_connection.size(3) - x.size(3)
        x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x, skip_connection], dim=1)
        x = self.lrgs_mamba(x)
        return x

class MambaMed2DUNet(nn.Module):
    def __init__(self, input_channels, num_classes, base_channels=32, deep_supervision=True, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        self.enc1 = CascadeEncoderFirst(input_channels, base_channels)
        self.enc2 = CascadeEncoder(base_channels, base_channels * 2)
        self.enc3 = CascadeEncoder(base_channels * 2, base_channels * 4)
        self.enc4 = CascadeEncoder(base_channels * 4, base_channels * 8)
        self.bottleneck = BottleneckBlock(base_channels * 8, base_channels * 16)
        self.dec4 = DecoderBlock(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = DecoderBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2 + base_channels, base_channels)

        self.main_output = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        if deep_supervision:
            self.ds_output4 = nn.Conv2d(base_channels * 8, num_classes, 1)
            self.ds_output3 = nn.Conv2d(base_channels * 4, num_classes, 1)
            self.ds_output2 = nn.Conv2d(base_channels * 2, num_classes, 1)

    def forward(self, x):
        e1, e1_pooled = self.enc1(x)
        e2, e2_pooled = self.enc2(e1_pooled)
        e3, e3_pooled = self.enc3(e2_pooled)
        e4, e4_pooled = self.enc4(e3_pooled)
        b = self.bottleneck(e4_pooled)
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        main_out = self.main_output(d1)
        if self.deep_supervision:
            ds4 = self.ds_output4(d4)
            ds3 = self.ds_output3(d3)
            ds2 = self.ds_output2(d2)
            return [main_out, ds2, ds3, ds4] if self.training else [main_out]
        else:
            return main_out

class nnUNetTrainerMambaMed(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        self.network_class = MambaMed2DUNet
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        print(f"✅ 使用【论文级联逻辑】的Mamba U-Net：前半部分双分支频域+超图，后半部分LRGS+MC-Mamba")

    def set_deep_supervision_enabled(self, enabled: bool):
        self.network.deep_supervision = enabled

    def on_train_start(self):
        super().on_train_start()
        if isinstance(self.network, MambaMed2DUNet):
            patch_size = self.configuration_manager.patch_size
            input_size = (1, self.num_input_channels, patch_size[0], patch_size[1])
            print("\n" + "=" * 80)
            print("📊 论文级联逻辑网络结构摘要（前半部分频域+超图，后半部分LRGS+MC-Mamba）")
            print("=" * 80)
            summary(self.network, input_size=input_size, verbose=1,
                    col_names=["input_size", "output_size", "num_params", "kernel_size"], depth=3)
        else:
            print("⚠️  警告：未使用论文级联逻辑的自定义网络！")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_num_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}