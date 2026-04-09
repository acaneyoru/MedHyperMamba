import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torchinfo import summary


class EnhancedWaveletTransform(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.register_buffer('ll_weight', torch.ones(1, 1, 2, 2) / 2)
        self.register_buffer('lh_weight', torch.tensor([[[[1, -1], [1, -1]]]], dtype=torch.float32) / 2)
        self.register_buffer('hl_weight', torch.tensor([[[[1, 1], [-1, -1]]]], dtype=torch.float32) / 2)
        self.register_buffer('hh_weight', torch.tensor([[[[1, -1], [-1, 1]]]], dtype=torch.float32) / 2)

        self.pwc = nn.Conv2d(channels, channels, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def dwt(self, x):
        b, c, h, w = x.shape
        x = x.view(b * c, 1, h, w)

        ll = F.conv2d(x, self.ll_weight.expand(c, 1, 2, 2), stride=2, groups=c)
        lh = F.conv2d(x, self.lh_weight.expand(c, 1, 2, 2), stride=2, groups=c)
        hl = F.conv2d(x, self.hl_weight.expand(c, 1, 2, 2), stride=2, groups=c)
        hh = F.conv2d(x, self.hh_weight.expand(c, 1, 2, 2), stride=2, groups=c)

        ll = ll.view(b, c, h // 2, w // 2)
        lh = lh.view(b, c, h // 2, w // 2)
        hl = hl.view(b, c, h // 2, w // 2)
        hh = hh.view(b, c, h // 2, w // 2)

        return ll, lh, hl, hh

    def forward(self, x):
        _, lh, hl, hh = self.dwt(x)
        x_h = lh + hl + hh

        p_c = self.sigmoid(self.pwc(self.gap(x_h) + self.gmp(x_h)))

        p = F.conv2d(p_c.expand_as(x_h),
                     torch.ones(self.channels, 1, 3, 3, device=x.device) / 9,
                     padding=1, groups=self.channels)

        attn = self.attention(torch.cat([x_h, p], dim=1))
        x_h_prime = x_h * attn + x_h

        lambda_thresh = torch.std(x_h_prime, dim=(2, 3), keepdim=True) * 0.5
        x_h_final = torch.where(
            x_h_prime > lambda_thresh,
            torch.sign(x_h_prime) * (torch.abs(x_h_prime) - lambda_thresh),
            torch.zeros_like(x_h_prime)
        )

        return x_h_final


class FastFourierTransform(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.conv_real = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_imag = nn.Conv2d(channels, channels, kernel_size=1)

        self.register_buffer('sigma', torch.tensor(3.0))

    def forward(self, x):
        b, c, h, w = x.shape

        x_fft = torch.fft.fft2(x)
        x_real = x_fft.real
        x_imag = x_fft.imag

        real_out = self.conv_real(x_real) - self.conv_imag(x_imag)
        imag_out = self.conv_real(x_imag) + self.conv_imag(x_real)
        x_fft_prime = torch.complex(real_out, imag_out)

        freq_y = torch.fft.fftfreq(h, device=x.device).view(-1, 1)
        freq_x = torch.fft.fftfreq(w, device=x.device).view(1, -1)
        gaussian_filter = torch.exp(-(freq_y ** 2 + freq_x ** 2) / (2 * self.sigma ** 2))
        gaussian_filter = gaussian_filter.view(1, 1, h, w)

        x_fft_filtered = x_fft_prime * gaussian_filter

        x_l_final = torch.fft.ifft2(x_fft_filtered).real

        return x_l_final


class FrequencyDomainDecomposition(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ewt = EnhancedWaveletTransform(channels)
        self.fft = FastFourierTransform(channels)

        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        x_high = self.ewt(x)
        x_low = self.fft(x)

        x_fused = self.fusion(torch.cat([x_high, x_low], dim=1))

        return x_high, x_low, x_fused


class HypergraphGuidedMultimodalFusion(nn.Module):
    def __init__(self, channels, num_modalities=4):
        super().__init__()
        self.channels = channels
        self.num_modalities = num_modalities

        self.freq_decomp = FrequencyDomainDecomposition(channels)

        self.map_proj = nn.Linear(channels * 2, channels)

        self.hgn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )

        self.encoder_zp = nn.Linear(channels, channels // 2)
        self.encoder_zq = nn.Linear(channels, channels // 2)
        self.decoder = nn.Linear(channels, channels)

        self.theta = nn.Parameter(torch.randn(3))

    def compute_information_entropy(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        probs = F.softmax(x_flat, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy.mean(dim=1, keepdim=True)

    def modal_wise_attention_pooling(self, features_list):
        shared_features = []

        for j in range(self.num_modalities):
            alpha_sum = 0
            v2_j = 0

            for k in range(self.num_modalities):
                sim = F.cosine_similarity(
                    features_list[j].flatten(1),
                    features_list[k].flatten(1),
                    dim=1
                ).unsqueeze(1)

                alpha_jk = torch.exp(sim)
                alpha_sum += alpha_jk

                v2_j += alpha_jk * features_list[k]

            v2_j = v2_j / (alpha_sum + 1e-8)
            shared_features.append(v2_j)

        return shared_features

    def construct_hypergraph(self, high_features, low_features, shared_features):
        V1 = high_features + low_features
        V2 = shared_features

        w_intra = []
        for i in range(self.num_modalities):
            h_entropy = self.compute_information_entropy(high_features[i])
            l_entropy = self.compute_information_entropy(low_features[i])
            w_intra.append((h_entropy + l_entropy) / 2)

        w_inter = []
        for p in range(len(V2)):
            for q in range(p + 1, len(V2)):
                attn_score = F.cosine_similarity(
                    V2[p].flatten(1), V2[q].flatten(1), dim=1
                ).mean()
                struct_sim = F.mse_loss(V2[p], V2[q], reduction='none').mean()
                w_inter.append(attn_score * torch.exp(-struct_sim))

        return V1, V2, w_intra, w_inter

    def hypergraph_convolution(self, V, E_weights):
        V_stacked = torch.stack([v.flatten(1) for v in V], dim=1)

        A = torch.ones(len(V), len(V), device=V_stacked.device) * torch.tensor(E_weights).view(-1, 1)
        D = torch.diag(A.sum(dim=1) + 1e-8)
        A_norm = torch.inverse(D) @ A

        Z = self.hgn(V_stacked @ A_norm)

        return Z

    def disentangle_factors(self, Z):
        Z_mean = Z.mean(dim=1)
        Z_p = self.encoder_zp(Z_mean)
        Z_q = self.encoder_zq(Z_mean)

        kl_loss = F.kl_div(
            F.log_softmax(Z_q, dim=-1),
            F.softmax(Z_p, dim=-1),
            reduction='batchmean'
        )

        omega = F.softmax(self.theta, dim=0)

        Z_S = self.decoder(torch.cat([Z_p, Z_q], dim=-1))

        return Z_S, kl_loss

    def forward(self, modalities):
        high_features = []
        low_features = []
        fused_features = []

        for modal in modalities:
            high, low, fused = self.freq_decomp(modal)
            high_features.append(high)
            low_features.append(low)
            fused_features.append(fused)

        shared_features = self.modal_wise_attention_pooling(fused_features)

        V1, V2, w_intra, w_inter = self.construct_hypergraph(
            high_features, low_features, shared_features
        )

        Z = self.hypergraph_convolution(V1 + V2, w_intra + w_inter)

        Z_S, kl_loss = self.disentangle_factors(Z)

        b, c, h, w = modalities[0].shape
        registered_features = Z_S.view(b, self.num_modalities, c, h, w).sum(dim=1)

        return registered_features, kl_loss


class SelectiveSSM(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(dim * expand)

        self.in_proj = nn.Linear(dim, self.d_inner * 2)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(d_state, self.d_inner)

        self.out_proj = nn.Linear(self.d_inner, dim)

        A = torch.arange(1, d_state + 1).float().view(1, -1)
        self.register_buffer('A_log', torch.log(A))
        self.register_buffer('D', nn.Parameter(torch.ones(self.d_inner)))

    def forward(self, x):
        b, l, d = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)

        x = F.silu(x)

        x_proj = self.x_proj(x)
        delta, B, C = x_proj.split([1, self.d_state, self.d_state], dim=-1)

        A = -torch.exp(self.A_log.float())
        delta = F.softplus(delta)

        dA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        dB = torch.einsum('bld,bldn->bldn', delta, B)

        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device)
        ys = []

        for i in range(l):
            h = h * dA[:, i] + dB[:, i].unsqueeze(1) * x[:, i].unsqueeze(-1)
            y = torch.einsum('bdn,bn->bd', h, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)

        y = y * F.silu(z)
        out = self.out_proj(y)

        return out


class LowRankHypergraphicScanning(nn.Module):
    def __init__(self, channels, patch_size=4):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size

        self.omega = nn.Parameter(torch.ones(3) / 3)

        self.hyper_attn = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid()
        )

        self.path_hgn = nn.Sequential(
            nn.Linear(channels + 4, channels),
            nn.GELU(),
            nn.Linear(channels, channels // 2)
        )

        self.delta = nn.Parameter(torch.tensor(0.1))

    def compute_grayscale_entropy(self, x, window_size=5):
        b, c, h, w = x.shape
        pad = window_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')

        entropy_map = torch.zeros(b, 1, h, w, device=x.device)

        for i in range(h):
            for j in range(w):
                patch = x_pad[:, :, i:i + window_size, j:j + window_size]
                patch_flat = patch.flatten(2)

                probs = F.softmax(patch_flat, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                entropy_map[:, :, i, j] = entropy.mean(dim=1, keepdim=True)

        return entropy_map

    def compute_noise_variance(self, x, window_size=5):
        kernel_size = 3
        sigma = 1.0
        kernel = torch.exp(-torch.arange(kernel_size).float() ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1).repeat(1, 1, 1)

        x_gaussian = F.conv2d(
            F.pad(x, (1, 1, 1, 1), mode='reflect'),
            kernel.view(1, 1, kernel_size, 1).repeat(x.shape[1], 1, 1, 1),
            groups=x.shape[1]
        )
        x_gaussian = F.conv2d(
            F.pad(x_gaussian, (1, 1, 1, 1), mode='reflect'),
            kernel.view(1, 1, 1, kernel_size).repeat(x.shape[1], 1, 1, 1),
            groups=x.shape[1]
        )

        noise_var = F.avg_pool2d((x - x_gaussian) ** 2, window_size, stride=1, padding=window_size // 2)

        return noise_var.mean(dim=1, keepdim=True)

    def compute_hyperedge_weight(self, x):
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        weights = self.hyper_attn(x_flat)
        return weights.reshape(b, 1, h, w)

    def compute_regional_information_weight(self, x):
        omega = F.softmax(self.omega, dim=0)

        H = self.compute_grayscale_entropy(x)
        H_norm = (H - H.mean()) / (H.std() + 1e-8)

        W_hyper = self.compute_hyperedge_weight(x)
        W_norm = (W_hyper - W_hyper.mean()) / (W_hyper.std() + 1e-8)

        N = self.compute_noise_variance(x)
        N_norm = 1 / (1 + torch.exp(-N))

        W_area = omega[0] * H_norm + omega[1] * W_norm + omega[2] * (1 - N_norm)

        return W_area

    def low_rank_redundancy_suppression(self, x, W_area):
        alpha = 0.5
        theta = alpha * W_area.max()
        M_valid = (W_area >= theta).float()

        x_denoise1 = torch.where(
            M_valid.bool(),
            x,
            F.avg_pool2d(
                F.pad(x, (1, 1, 1, 1), mode='reflect'),
                3, stride=1
            )
        )

        x_crop = x_denoise1 * M_valid

        tau = self.compute_noise_variance(x_crop).sqrt()
        x_denoise2 = torch.where(
            x_crop > tau,
            x_crop - tau,
            torch.where(
                x_crop < -tau,
                x_crop + tau,
                torch.zeros_like(x_crop)
            )
        )

        return x_denoise2, M_valid

    def hypergraph_guided_path_planning(self, x, W_area, M_valid):
        b, c, h, w = x.shape

        ph, pw = h // self.patch_size, w // self.patch_size
        patches = x.view(b, c, ph, self.patch_size, pw, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(b, ph * pw, c, self.patch_size * self.patch_size)

        v_raw = torch.cat([
            patches.mean(dim=-1),
            patches.max(dim=-1)[0]
        ], dim=-1)
        v_raw = F.linear(v_raw, torch.eye(c * 2, c, device=x.device))

        gamma = 0.1
        W_patch = F.adaptive_avg_pool2d(W_area, (ph, pw)).view(b, -1, 1)
        N_patch = F.adaptive_avg_pool2d(
            self.compute_noise_variance(x), (ph, pw)
        ).view(b, -1, 1)

        S_k = W_patch * (1 - gamma * N_patch)

        M_patch = F.adaptive_avg_pool2d(M_valid, (ph, pw)).view(b, -1)
        valid_indices = M_patch > 0.5

        spatial_dist = torch.cdist(
            torch.stack(torch.meshgrid(
                torch.arange(ph, device=x.device),
                torch.arange(pw, device=x.device)
            ), dim=-1).float().view(-1, 2),
            torch.stack(torch.meshgrid(
                torch.arange(ph, device=x.device),
                torch.arange(pw, device=x.device)
            ), dim=-1).float().view(-1, 2)
        )

        tau_hyper = torch.exp(-self.delta * spatial_dist)
        tau_priority = S_k.squeeze(-1)

        path_logits = tau_hyper * tau_priority.unsqueeze(1)

        scan_order = torch.argsort(path_logits.sum(dim=1), descending=True)

        return scan_order, v_raw, valid_indices

    def forward(self, x):
        W_area = self.compute_regional_information_weight(x)

        x_denoised, M_valid = self.low_rank_redundancy_suppression(x, W_area)

        scan_order, v_raw, valid_indices = self.hypergraph_guided_path_planning(
            x_denoised, W_area, M_valid
        )

        b, c, h, w = x.shape
        ph, pw = h // self.patch_size, w // self.patch_size

        x_flat = x_denoised.view(b, c, ph, self.patch_size, pw, self.patch_size)
        x_flat = x_flat.permute(0, 2, 4, 1, 3, 5).contiguous()
        x_flat = x_flat.view(b, ph * pw, c * self.patch_size * self.patch_size)

        scanned_sequence = torch.stack([
            x_flat[b_idx, scan_order[b_idx]]
            for b_idx in range(b)
        ])

        scanned_sequence = F.linear(
            scanned_sequence,
            torch.eye(c * self.patch_size * self.patch_size, c, device=x.device)
        )

        return scanned_sequence, scan_order, (h, w, ph, pw)


class LocalEnhancementMamba(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.pos_mlp = nn.Sequential(
            nn.Linear(4, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(dim)

    def get_relative_position_encoding(self, h, w, device):
        coords_h = torch.arange(h, device=device)
        coords_w = torch.arange(w, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)
        coords_flat = coords.reshape(-1, 2)

        rel_pos = coords_flat.unsqueeze(1) - coords_flat.unsqueeze(0)
        pos_enc = torch.cat([rel_pos, rel_pos.abs()], dim=-1).float()

        return self.pos_mlp(pos_enc)

    def window_partition(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h // self.window_size, self.window_size,
                   w // self.window_size, self.window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return windows.view(-1, c, self.window_size, self.window_size)

    def window_reverse(self, windows, h, w):
        b = int(windows.shape[0] / (h * w / self.window_size ** 2))
        x = windows.view(b, h // self.window_size, w // self.window_size,
                         -1, self.window_size, self.window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(b, -1, h, w)

    def forward(self, x, hyperedge_weights=None):
        b, c, h, w = x.shape
        shortcut = x

        ca = self.channel_attn(x)
        x = x * ca

        x_windows = self.window_partition(x)
        nw, c, wh, ww = x_windows.shape

        x_flat = x_windows.view(nw, c, -1).transpose(1, 2)

        q = self.W_q(x_flat).view(nw, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x_flat).view(nw, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x_flat).view(nw, -1, self.num_heads, self.head_dim).transpose(1, 2)

        pos_enc = self.get_relative_position_encoding(wh, ww, x.device)
        pos_enc = pos_enc.view(wh * ww, wh * ww, self.dim)
        pos_bias = (q @ pos_enc.transpose(-1, -2).to(q.dtype)).mean(dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale + pos_bias
        attn = F.softmax(attn, dim=-1)

        if hyperedge_weights is not None:
            attn = attn * hyperedge_weights.view(nw, 1, 1, -1)

        x_attn = (attn @ v).transpose(1, 2).reshape(nw, -1, c)
        x_attn = x_attn.transpose(1, 2).view(nw, c, wh, ww)

        x_local = self.window_reverse(x_attn, h, w)

        x_out = F.layer_norm(
            (shortcut + x_local).permute(0, 2, 3, 1),
            (c,)
        ).permute(0, 3, 1, 2)

        return x_out


class GlobalAssociationMamba(nn.Module):
    def __init__(self, dim, d_state=16, num_modalities=4):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.num_modalities = num_modalities

        self.ssm = SelectiveSSM(dim, d_state)

        self.W_h = nn.Linear(dim, dim)
        self.W_x = nn.Linear(dim, dim)
        self.W_g = nn.Linear(dim, dim)

        self.W_gate_h = nn.Linear(dim, dim)
        self.W_gate_s = nn.Linear(dim, dim)

        self.modal_proj = nn.Linear(dim, dim)
        self.attn_proj = nn.Linear(dim, dim)
        self.fusion_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, modalities_features=None):
        b, l, c = x.shape

        S_bar = x.mean(dim=1, keepdim=True)

        h = torch.zeros(b, self.dim, device=x.device)
        outputs = []

        for t in range(l):
            S_t = x[:, t]

            gate = torch.sigmoid(self.W_gate_h(h) + self.W_gate_s(S_t))

            h = F.gelu(
                self.W_h(h) + self.W_x(S_t) + self.W_g(S_bar.squeeze(1)) + gate
            )
            outputs.append(h.unsqueeze(1))

        h_seq = torch.cat(outputs, dim=1)

        h_ssm = self.ssm(x)

        if modalities_features is not None:
            F_m = torch.stack([
                self.modal_proj(mf.mean(dim=(2, 3)).view(b, -1))
                for mf in modalities_features
            ], dim=1)

            F_bar = F_m.mean(dim=1, keepdim=True)

            alpha = F.softmax(
                torch.einsum('bmd,bd->bm', F_m, self.attn_proj(F_bar.squeeze(1))),
                dim=1
            )

            h_fusion = h_seq + torch.einsum('bm,bmd->bd', alpha, F_m).unsqueeze(1)
        else:
            h_fusion = h_seq

        x_global = self.norm(x + h_fusion + h_ssm)

        return x_global


class MultiScaleFusionMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.conv_s2 = nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim)
        self.conv_s4 = nn.Conv2d(dim, dim, 3, stride=4, padding=1, groups=dim)
        self.conv_s1 = nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim)

        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.proj3 = nn.Conv2d(dim, dim, 1)
        self.proj4 = nn.Conv2d(dim, dim, 1)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.final_conv = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, x, hyperedge_weights=None):
        b, c, h, w = x.shape

        X1 = self.conv_s1(x)
        X2 = self.conv_s2(x)
        X3 = self.conv_s4(x)
        X4 = F.interpolate(x, scale_factor=2, mode='bilinear')

        scales = [X1, X2, X3, X4]
        sizes = [(h, w), (h // 2, w // 2), (h // 4, w // 4), (h * 2, w * 2)]

        projs = [self.proj1, self.proj2, self.proj3, self.proj4]
        X_proj = [proj(X) for proj, X in zip(projs, scales)]

        F_i = [F.adaptive_avg_pool2d(X, 1) for X in X_proj]

        M = torch.zeros(4, 4, device=x.device)
        for i in range(4):
            for j in range(4):
                f_i = F_i[i].flatten(1)
                f_j = F_i[j].flatten(1)
                omega_ij = 1.0 if hyperedge_weights is None else hyperedge_weights.mean()
                M[i, j] = (f_i @ f_j.t() + omega_ij).mean()

        M = F.softmax(M, dim=1)

        X_fusion = []
        for i in range(4):
            fused = X_proj[i]
            for j in range(4):
                if i != j:
                    Xj_resized = F.interpolate(X_proj[j], size=sizes[i], mode='bilinear')
                    fused = fused + M[i, j] * Xj_resized
            X_fusion.append(fused)

        X_enhanced = []
        for X_f in X_fusion:
            ca = self.channel_attn(X_f)
            X_e = F.layer_norm(
                (X_f + self.conv3x3(X_f * ca)).permute(0, 2, 3, 1),
                (c,)
            ).permute(0, 3, 1, 2)
            X_enhanced.append(X_e)

        target_size = sizes[0]
        X_aligned = []
        for X_e, size in zip(X_enhanced, sizes):
            if size != target_size:
                X_aligned.append(F.interpolate(X_e, size=target_size, mode='bilinear'))
            else:
                X_aligned.append(X_e)

        X_concat = torch.cat(X_aligned, dim=1)
        X_multi = self.final_conv(X_concat)

        return X_multi


class MCMambaBlock(nn.Module):
    def __init__(self, dim, window_size=8, num_modalities=4):
        super().__init__()
        self.le_mamba = LocalEnhancementMamba(dim, window_size)
        self.ga_mamba = GlobalAssociationMamba(dim, num_modalities=num_modalities)
        self.msf_mamba = MultiScaleFusionMamba(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, modalities_features=None, hyperedge_weights=None):
        b, c, h, w = x.shape

        x_local = self.le_mamba(x, hyperedge_weights)

        x_seq = x_local.permute(0, 2, 3, 1).view(b, h * w, c)
        x_global = self.ga_mamba(x_seq, modalities_features)
        x_global = x_global.view(b, h, w, c).permute(0, 3, 1, 2)

        x_multi = self.msf_mamba(x_global, hyperedge_weights)

        return x_multi


class MedHyperMamba2D(nn.Module):
    def __init__(self, input_channels, num_classes, base_channels=32,
                 deep_supervision=True, **kwargs):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.base_channels = base_channels

        self.num_modalities = 4
        assert input_channels == self.num_modalities, "MedHyperMamba expects 4 input modalities"

        self.stem = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

        self.hgmf = HypergraphGuidedMultimodalFusion(base_channels, self.num_modalities)

        self.lrhs = LowRankHypergraphicScanning(base_channels)

        self.enc1 = self._make_encoder_block(base_channels, base_channels * 2)
        self.enc2 = self._make_encoder_block(base_channels * 2, base_channels * 4)
        self.enc3 = self._make_encoder_block(base_channels * 4, base_channels * 8)
        self.enc4 = self._make_encoder_block(base_channels * 8, base_channels * 16)

        self.mcmamba1 = MCMambaBlock(base_channels)
        self.mcmamba2 = MCMambaBlock(base_channels * 2)
        self.mcmamba3 = MCMambaBlock(base_channels * 4)
        self.mcmamba4 = MCMambaBlock(base_channels * 8)
        self.mcmamba_bottleneck = MCMambaBlock(base_channels * 16)

        self.dec4 = self._make_decoder_block(base_channels * 16, base_channels * 8)
        self.dec3 = self._make_decoder_block(base_channels * 8, base_channels * 4)
        self.dec2 = self._make_decoder_block(base_channels * 4, base_channels * 2)
        self.dec1 = self._make_decoder_block(base_channels * 2, base_channels)

        self.main_output = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        if deep_supervision:
            self.ds_output4 = nn.Conv2d(base_channels * 8, num_classes, kernel_size=1)
            self.ds_output3 = nn.Conv2d(base_channels * 4, num_classes, kernel_size=1)
            self.ds_output2 = nn.Conv2d(base_channels * 2, num_classes, kernel_size=1)

        self.kl_weight = 0.001

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        modalities = [x[:, i:i + 1] for i in range(self.num_modalities)]

        modal_features = [self.stem(m) for m in modalities]

        registered_features, kl_loss = self.hgmf(modal_features)

        scanned_seq, scan_order, (sh, sw, ph, pw) = self.lrhs(registered_features)

        e0 = registered_features

        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        e0_mamba = self.mcmamba1(e0, modal_features)
        e1_mamba = self.mcmamba2(e1)
        e2_mamba = self.mcmamba3(e2)
        e3_mamba = self.mcmamba4(e3)
        b_mamba = self.mcmamba_bottleneck(e4)

        e0 = e0 + e0_mamba
        e1 = e1 + e1_mamba
        e2 = e2 + e2_mamba
        e3 = e3 + e3_mamba
        e4 = e4 + b_mamba

        d4 = self.dec4(torch.cat([e4, e3], dim=1))
        d3 = self.dec3(torch.cat([d4, e2], dim=1))
        d2 = self.dec2(torch.cat([d3, e1], dim=1))
        d1 = self.dec1(torch.cat([d2, e0], dim=1))

        main_out = self.main_output(d1)

        self.kl_loss = kl_loss

        if self.deep_supervision:
            ds4 = self.ds_output4(d4)
            ds3 = self.ds_output3(d3)
            ds2 = self.ds_output2(d2)
            return main_out, ds4, ds3, ds2
        else:
            return main_out

class nnUNetTrainerMyMed(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        self.network_class = MedHyperMamba2D
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    def set_deep_supervision_enabled(self, enabled: bool):
        self.network.deep_supervision = enabled

    def on_train_start(self):
        super().on_train_start()

        if isinstance(self.network, MedHyperMamba2D):
            patch_size = self.configuration_manager.patch_size
            input_size = (1, self.num_input_channels, patch_size[0], patch_size[1])

    def train_step(self, batch):
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        output = self.network(data)

        if isinstance(output, list):
            loss = self.loss(output[0], target[0] if isinstance(target, list) else target)
            for i in range(1, len(output)):
                loss += 0.5 ** (len(output) - i) * self.loss(output[i],
                                                             target[i] if isinstance(target, list) else target)
        else:
            loss = self.loss(output, target)

        if hasattr(self.network, 'kl_loss'):
            loss = loss + self.network.kl_loss * 0.001

        loss.backward()

        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return loss.detach()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6
        )

        return optimizer, scheduler
