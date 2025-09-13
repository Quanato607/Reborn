#!/usr/bin/env python3
# encoding: utf-8
"""
HiKINet — complete, robust implementation for multi-modal 3D MRI segmentation
with selectable per-level fusion (concat / MoE / PoE), missing-branch
bottleneck re-fusion (MoE / PoE), and KD (MSE/KL/Cosine) from fused features to
hallucinated (fake) features.

Key features
- BasicBlock: GroupNorm-preact residual block with auto residual projection
- Encoder: 4-stage UNet-like encoder with style/content mixing at bottleneck
- Decoder: UNet-like decoder with segmentation head and projection head
- MoE: spatial gating with expert masking & renormalization
- PoE: diagonal-Gaussian product with expert masking & safe fallback
- Missing-branch: re-fusion at bottleneck (fus_c4d vs fake_c4d) via MoE/PoE
- KD: shared 1x1 head, supports "mse"(default) / "kl" / "cosine"; KL uses strict no-grad teacher

Author: (your name)
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Basic building blocks
# ----------------------------
class DWConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv3d(in_ch, in_ch, kernel_size=kernel_size, padding=padding,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
    def forward(self, x):
        return self.pw(self.dw(x))

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8, use_dw=True):
        super().__init__()
        g1 = n_groups if in_channels % n_groups == 0 else 1
        self.gn1 = nn.GroupNorm(g1, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        if use_dw:
            self.conv1 = DWConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.res_conv = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
    def forward(self, x):
        residual = self.res_conv(x)
        x = self.conv1(self.relu1(self.gn1(x)))
        return x + residual



class Encoder(nn.Module):
    """UNet-like 3D encoder with style-content mixing at the bottleneck.
    Ref: Myronenko, "3D MRI brain tumor segmentation using autoencoder regularization"
    """

    def __init__(self, in_channels: int = 1, init_channels: int = 32, p: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.init_channels = init_channels
        self._make_encoder()
        self.dropout = nn.Dropout(p=p)

    def _make_encoder(self) -> None:
        C = self.init_channels
        # Stage 1
        self.conv1a = nn.Conv3d(self.in_channels, C, kernel_size=3, padding=1)
        self.conv1b = BasicBlock(C, C)
        self.ds1 = nn.Conv3d(C, C * 2, kernel_size=3, stride=2, padding=1)  # -> 2C

        # Stage 2
        self.conv2a = BasicBlock(C * 2, C * 2)
        self.ds2 = nn.Conv3d(C * 2, C * 4, kernel_size=3, stride=2, padding=1)  # -> 4C

        # Stage 3
        self.conv3a = BasicBlock(C * 4, C * 4)
        self.ds3 = nn.Conv3d(C * 4, C * 8, kernel_size=3, stride=2, padding=1)  # -> 8C

        # Stage 4 (bottleneck)
        self.conv4a = BasicBlock(C * 8, C * 8)

        # Style/content mixers
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)  # c2d -> c2d_p (/8)
        self.convc = nn.Conv3d(C * 20, C * 8, kernel_size=1)   # [c2d_p(4C)|c3d(8C)|c4d(8C)] -> 8C
        self.convco = nn.Conv3d(C * 16, C * 8, kernel_size=1)  # [style(8C)|content(8C)] -> 8C

    def forward(self, x: torch.Tensor):
        # enc1
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)  # (B, C, D, H, W)
        c1d = self.ds1(c1)    # (B, 2C, D/2, H/2, W/2)

        # enc2
        c2 = self.conv2a(c1d)
        c2d = self.ds2(c2)    # (B, 4C, /4)
        c2d_p = self.pool(c2d)  # (B, 4C, /8)

        # enc3
        c3 = self.conv3a(c2d)
        c3d = self.ds3(c3)    # (B, 8C, /8)

        # enc4
        c4d = self.conv4a(c3d)
        # style-content fusion at bottleneck
        style = self.convc(torch.cat([c2d_p, c3d, c4d], dim=1))  # (B, 8C, /8)
        content = c4d
        c4d = self.convco(torch.cat([style, content], dim=1))    # (B, 8C, /8)
        c4d = self.dropout(c4d)
        return c4d


class Decoder(nn.Module):
    """UNet-like 3D decoder with segmentation and projection heads."""

    def __init__(self, out_channels: int = 3, init_channels: int = 32, embed_dim: int = 128, proj_dim: int = 64, p: float = 0.2, seg_activation: str = "sigmoid"):
        """
        seg_activation: "sigmoid" | "softmax" | "none"
            - 若训练使用 *WithLogits* 的损失（如 BCEWithLogits/DiceWithLogits），建议设为 "none"
        """
        super().__init__()
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.seg_activation = seg_activation.lower()
        assert self.seg_activation in {"sigmoid", "softmax", "none"}

        self._make_decoder()
        self.dropout = nn.Dropout(p=p)

        # segmentation head: (B, C, D, H, W) -> (B, out_channels, D, H, W)
        self.seg_conv = nn.Conv3d(init_channels, out_channels, kernel_size=1)

        # projection head: global embedding
        self.proj_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # (B,C,1,1,1)
            nn.Flatten(1),
            nn.Linear(init_channels, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

    def _make_decoder(self) -> None:
        C = self.init_channels
        self.up4conva = nn.Conv3d(C * 8, C * 4, kernel_size=1)
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up4convb = BasicBlock(C * 4, C * 4)

        self.up3conva = nn.Conv3d(C * 4, C * 2, kernel_size=1)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up3convb = BasicBlock(C * 2, C * 2)

        self.up2conva = nn.Conv3d(C * 2, C, kernel_size=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up2convb = BasicBlock(C, C)

    def forward(self, c4d: torch.Tensor):
        x = self.dropout(c4d)

        x = self.up4conva(x)  # (B,4C,/8)
        x = self.up4(x)       # (B,4C,/4)
        x = self.up4convb(x)

        x = self.up3conva(x)  # (B,2C,/4)
        x = self.up3(x)       # (B,2C,/2)
        x = self.up3convb(x)

        x = self.up2conva(x)  # (B,C,/2)
        x = self.up2(x)       # (B,C,/1)
        x = self.up2convb(x)

        seg_logits = self.seg_conv(x)
        if self.seg_activation == "sigmoid":
            seg_out = torch.sigmoid(seg_logits)
        elif self.seg_activation == "softmax":
            seg_out = torch.softmax(seg_logits, dim=1)
        else:
            seg_out = seg_logits  # no activation (for *WithLogits losses)

        proj_out = self.proj_head(x)
        return proj_out, seg_out


# ----------------------------
# MoE / PoE helpers
# ----------------------------
class MoEGate(nn.Module):
    """Spatial MoE gate producing per-expert weights (sum to 1 across experts).
    Robust to zero-expert inputs via masking + renormalization.
    """

    def __init__(self, in_ch: int, num_experts: int = 4, temperature: float = 1.0):
        super().__init__()
        self.logits = nn.Conv3d(in_ch, num_experts, kernel_size=1)
        self.temperature = temperature
        self.num_experts = num_experts

    def forward(self, feats_list):
        x = torch.cat(feats_list, dim=1)
        g = self.logits(x) / max(self.temperature, 1e-6)
        w = torch.softmax(g, dim=1)  # (B, E, D, H, W)

        # zero-expert robustness: per-voxel valid mask & renormalize
        valid = expert_valid_mask(feats_list)  # (B,E,D,H,W) in {0,1}
        w = w * valid
        sum_w = w.sum(dim=1, keepdim=True)  # (B,1,D,H,W)
        uniform = w.new_full(w.shape, 1.0 / self.num_experts)
        all_invalid = (sum_w <= 0)
        w_norm = w / (sum_w + 1e-6)
        w = torch.where(all_invalid.expand_as(w), uniform, w_norm)
        return w


def expert_valid_mask(feats_list, eps: float = 1e-8):
    """Per-expert, per-voxel validity mask.
    A voxel is valid for an expert if average |activation| across channels > eps.
    Returns float mask (B, E, D, H, W) with values in {0,1}.
    """
    mags = [f.abs().mean(dim=1, keepdim=True) for f in feats_list]  # list of (B,1,D,H,W)
    m = torch.stack(mags, dim=1).squeeze(2)  # (B,E,D,H,W)
    return (m > eps).float()


def fuse_moe(weights: torch.Tensor, feats_list):
    """weights: (B, E, D, H, W); feats_list: list of E tensors each (B, C, D, H, W)."""
    stack = torch.stack(feats_list, dim=1)  # (B,E,C,D,H,W)
    w = weights.unsqueeze(2)                # (B,E,1,D,H,W)
    fused = (w * stack).sum(dim=1)          # (B,C,D,H,W)
    return fused


class PoEHead(nn.Module):
    """Outputs (mu, logvar) for a Gaussian expert at a given level."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.param = nn.Conv3d(in_ch, in_ch * 2, kernel_size=1)

    def forward(self, f: torch.Tensor):
        C = f.shape[1]
        h = self.param(f)
        mu, logvar = torch.split(h, C, dim=1)
        logvar = torch.clamp(logvar, -5.0, 5.0)  # stabilize precisions
        return mu, logvar


def fuse_poe(mus, logvars, valid=None, eps: float = 1e-6):
    """Product-of-Experts fusion for diagonal Gaussians with optional expert masking.
    mus/logvars: lists of tensors of shape (B, C, D, H, W).
    valid: optional mask (B, E, D, H, W) in {0,1}.
    Returns fused mean (same shape as a single mu).
    """
    precisions = [torch.exp(-lv) for lv in logvars]
    num = None
    den = None
    for i, (mu, pr) in enumerate(zip(mus, precisions)):
        if valid is not None:
            vi = valid[:, i:i+1]  # (B,1,D,H,W)
            pr = pr * vi  # zero out invalid voxels for this expert
        term = pr * mu
        num = term if num is None else num + term
        den = pr if den is None else den + pr
    fused = num / (den + eps)

    if valid is not None:
        all_invalid = (valid.sum(dim=1, keepdim=True) <= 0)  # (B,1,D,H,W)
        mean_mu = torch.stack(mus, dim=1).mean(dim=1)
        fused = torch.where(all_invalid.expand_as(fused), mean_mu, fused)
    return fused


# ----------------------------
# Main network
# ----------------------------
class HiKINet(nn.Module):
    """Multi-branch 3D UNet for 4-modality input with configurable fusion.

    Args:
        input_shape: (D, H, W), unused but kept for compatibility.
        in_channels: number of modalities (4 for FLAIR, T1ce, T1, T2).
        out_channels: number of segmentation output channels.
        init_channels: base channels (C).
        p: dropout rate.
        fusion_mode: {"concat","moe","poe"} for per-level multi-modal fusion.
        missing_refusion_mode: {"moe","poe"} for bottleneck fusion between
            fus_c4d and fake_c4d under missing training.
        gate_temp: softmax temperature for MoE gates.
        kd_dim: channel dim for KD head producing per-voxel projections.
        kd_temp: temperature for KL distillation.
        kd_weight: scaling for the KD loss.
        kd_loss_type: {"mse","kl","cosine"} KD objective (default "mse").
        seg_activation: {"sigmoid","softmax","none"} activation in decoders.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        in_channels: int = 4,
        out_channels: int = 3,
        init_channels: int = 32,
        p: float = 0.2,
        fusion_mode: str = "poe",
        gate_temp: float = 1.0,
        missing_refusion_mode: str = "poe",
        kd_dim: int = 64,
        kd_temp: float = 1.0,
        kd_weight: float = 1.0,
        kd_loss_type: str = "mse",
        seg_activation: str = "sigmoid",
    ):
        super().__init__()
        assert in_channels == 4, "This implementation assumes 4 modalities: [FLAIR, T1ce, T1, T2]."
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.fusion_mode = fusion_mode.lower()
        assert self.fusion_mode in {"concat", "moe", "poe"}
        self.missing_refusion_mode = missing_refusion_mode.lower()
        assert self.missing_refusion_mode in {"moe", "poe"}
        self.kd_temp = kd_temp
        self.kd_weight = kd_weight
        self.kd_loss_type = kd_loss_type.lower()
        assert self.kd_loss_type in {"kl", "mse", "cosine"}
        self.seg_activation = seg_activation

        # per-modality encoders
        self.flair_encoder = Encoder(in_channels=1, init_channels=init_channels, p=p)
        self.t1ce_encoder = Encoder(in_channels=1, init_channels=init_channels, p=p)
        self.t1_encoder = Encoder(in_channels=1, init_channels=init_channels, p=p)
        self.t2_encoder = Encoder(in_channels=1, init_channels=init_channels, p=p)

        # hallucinated/fake encoder takes all modalities masked together
        self.fake_encoder = Encoder(in_channels=4, init_channels=init_channels, p=p)

        C = init_channels
        # Concat fusion heads (used only in concat mode)
        self.convc4 = nn.Conv3d(C * 32, C * 8, kernel_size=1)  # [4x(8C)] -> 8C
        self.convc3 = nn.Conv3d(C * 16, C * 4, kernel_size=1)  # [4x(4C)] -> 4C
        self.convc2 = nn.Conv3d(C * 8, C * 2, kernel_size=1)   # [4x(2C)] -> 2C
        self.convc  = nn.Conv3d(C * 4, C * 1, kernel_size=1)   # [4x(1C)] -> 1C

        # MoE gates (used only in moe mode)
        self.gate1 = MoEGate(in_ch=4 * C,     num_experts=4, temperature=gate_temp)
        self.gate2 = MoEGate(in_ch=4 * 2 * C, num_experts=4, temperature=gate_temp)
        self.gate3 = MoEGate(in_ch=4 * 4 * C, num_experts=4, temperature=gate_temp)
        self.gate4 = MoEGate(in_ch=4 * 8 * C, num_experts=4, temperature=gate_temp)

        # PoE heads (used only in poe mode) -> 4 experts per level
        self.poe1 = nn.ModuleList([PoEHead(C)     for _ in range(4)])
        self.poe2 = nn.ModuleList([PoEHead(2 * C) for _ in range(4)])
        self.poe3 = nn.ModuleList([PoEHead(4 * C) for _ in range(4)])
        self.poe4 = nn.ModuleList([PoEHead(8 * C) for _ in range(4)])

        # Bottleneck refusion for missing training (fus_c4d vs fake_c4d)
        self.bottleneck_gate = MoEGate(in_ch=2 * 8 * C, num_experts=2, temperature=gate_temp)
        self.poe_bottleneck  = nn.ModuleList([PoEHead(8 * C) for _ in range(2)])  # two experts: fus & fake

        # KD head (shared for teacher/student to same semantic space)
        self.kd_head = nn.Conv3d(8 * C, kd_dim, kernel_size=1)

        # decoders for full & missing inputs
        self.decoder_f = Decoder(out_channels=out_channels, init_channels=init_channels, p=p, seg_activation=self.seg_activation)
        self.decoder_m = Decoder(out_channels=out_channels, init_channels=init_channels, p=p, seg_activation=self.seg_activation)

    # -------- per-level fusion --------
    def _fuse_level(self, feats_list, level: int):
        """feats_list: [flair, t1ce, t1, t2] features at a given level.
        level: 1,2,3,4 (4 means c4d/bottleneck).
        Returns fused features with the same channel count as a single expert at that level.
        Robust to zero-expert inputs.
        """
        if self.fusion_mode == "concat":
            x = torch.cat(feats_list, dim=1)
            if   level == 1: return self.convc(x)
            elif level == 2: return self.convc2(x)
            elif level == 3: return self.convc3(x)
            elif level == 4: return self.convc4(x)
            else: raise ValueError("level must be in {1,2,3,4}")

        elif self.fusion_mode == "moe":
            if   level == 1: w = self.gate1(feats_list)
            elif level == 2: w = self.gate2(feats_list)
            elif level == 3: w = self.gate3(feats_list)
            elif level == 4: w = self.gate4(feats_list)
            else: raise ValueError("level must be in {1,2,3,4}")
            return fuse_moe(w, feats_list)

        elif self.fusion_mode == "poe":
            valid = expert_valid_mask(feats_list)
            if   level == 1: mus, lvs = zip(*[head(f) for head, f in zip(self.poe1, feats_list)])
            elif level == 2: mus, lvs = zip(*[head(f) for head, f in zip(self.poe2, feats_list)])
            elif level == 3: mus, lvs = zip(*[head(f) for head, f in zip(self.poe3, feats_list)])
            elif level == 4: mus, lvs = zip(*[head(f) for head, f in zip(self.poe4, feats_list)])
            else: raise ValueError("level must be in {1,2,3,4}")
            return fuse_poe(mus, lvs, valid=valid)

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    # -------- full-modality path (teacher) --------
    def forward_full(self, x: torch.Tensor, mask_tensor: torch.Tensor):
        # x: (B, 4, D, H, W), mask_tensor: broadcastable to x
        x_m = x * mask_tensor

        flair_c4d = self.flair_encoder(x[:, 0:1])
        t1ce_c4d = self.t1ce_encoder(x[:, 1:2])
        t1_c4d  = self.t1_encoder(x[:, 2:3])
        t2_c4d  = self.t2_encoder(x[:, 3:4])

        # fus_c1 = self._fuse_level([flair_c1, t1ce_c1, t1_c1, t2_c1], level=1)
        # fus_c2 = self._fuse_level([flair_c2, t1ce_c2, t1_c2, t2_c2], level=2)
        # fus_c3 = self._fuse_level([flair_c3, t1ce_c3, t1_c3, t2_c3], level=3)
        fus_c4d = self._fuse_level([flair_c4d, t1ce_c4d, t1_c4d, t2_c4d], level=4)

        # fake encoder reads masked inputs
        fake_c4d = self.fake_encoder(x_m)

        # Decode (teacher path)
        proj_f, seg_f = self.decoder_f(fus_c4d)

        # KD from teacher (fus_c4d) to student (fake_c4d)
        kd_full = self._kd_loss(fus_c4d, fake_c4d)

        return proj_f, seg_f, fus_c4d, fake_c4d, kd_full

    # -------- missing-modality path (student) --------
    def forward_missing(self, xm: torch.Tensor):
        # xm: masked input modalities (B,4,D,H,W)
        flair_c4d = self.flair_encoder(xm[:, 0:1])
        t1ce_c4d = self.t1ce_encoder(xm[:, 1:2])
        t1_c4d  = self.t1_encoder(xm[:, 2:3])
        t2_c4d  = self.t2_encoder(xm[:, 3:4])

        fus_c4d = self._fuse_level([flair_c4d, t1ce_c4d, t1_c4d, t2_c4d], level=4)

        # fake path
        fake_c4d = self.fake_encoder(xm)

        # Re-fuse bottleneck between fus and fake -> new c4d (c4d_m)
        c4d_m = self._refuse_bottleneck(fus_c4d, fake_c4d)

        proj_m, seg_m = self.decoder_m(c4d_m)

        return proj_m, seg_m, c4d_m

    # -------- utilities --------
    def _refuse_bottleneck(self, fus_c4d: torch.Tensor, fake_c4d: torch.Tensor) -> torch.Tensor:
        if self.missing_refusion_mode == "moe":
            w = self.bottleneck_gate([fus_c4d, fake_c4d])  # (B,2,D,H,W)
            return fuse_moe(w, [fus_c4d, fake_c4d])
        elif self.missing_refusion_mode == "poe":
            (mu_fus,  lv_fus)  = self.poe_bottleneck[0](fus_c4d)
            (mu_fake, lv_fake) = self.poe_bottleneck[1](fake_c4d)
            return fuse_poe([mu_fus, mu_fake], [lv_fus, lv_fake])
        else:
            raise ValueError(f"Unknown missing_refusion_mode: {self.missing_refusion_mode}")

    def _kd_loss(self, teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
        """
        KD objectives:
          - "kl": KL(student||teacher) in kd_head space with temperature T; teacher prob is no-grad
          - "mse": MSE between kd_head(teacher) and kd_head(student), teacher is no-grad
          - "cosine": 1 - cosine similarity in kd_head space
        """
        if self.kd_weight == 0.0:
            return torch.tensor(0.0, device=student.device, dtype=student.dtype)

        if self.kd_loss_type == "kl":
            s_logit = self.kd_head(student)
            T = max(self.kd_temp, 1e-6)
            with torch.no_grad():
                t_prob = F.softmax(self.kd_head(teacher) / T, dim=1)
            loss = F.kl_div(
                F.log_softmax(s_logit / T, dim=1),
                t_prob,
                reduction="batchmean",
            ) * (T * T)

        elif self.kd_loss_type == "mse":
            with torch.no_grad():
                t_feat = self.kd_head(teacher)
            s_feat = self.kd_head(student)
            loss = F.mse_loss(s_feat, t_feat)

        else:  # cosine
            with torch.no_grad():
                t_feat = self.kd_head(teacher)
            s_feat = self.kd_head(student)
            B, C, D, H, W = s_feat.shape
            t = t_feat.view(B, C, -1)
            s = s_feat.view(B, C, -1)
            cos = F.cosine_similarity(s, t, dim=1)  # (B, N)
            loss = (1 - cos).mean()

        return loss * self.kd_weight

    # -------- joint forward --------
    def forward(self, x: torch.Tensor, xm: torch.Tensor, mask_tensor: torch.Tensor):
        """Joint forward for full-modality and missing-modality branches.
        Returns:
            proj_f, seg_f, proj_m, seg_m, fus_c4d, c4d_m, kd_full
        """
        proj_f, seg_f, fus_c4d, fake_c4d, kd_full = self.forward_full(x, mask_tensor)
        proj_m, seg_m, c4d_m = self.forward_missing(xm)
        return proj_f, seg_f, proj_m, seg_m, fus_c4d, c4d_m, kd_full


# Optional smoke test
if __name__ == "__main__":
    B, C, D, H, W = 1, 4, 32, 32, 32
    x  = torch.randn(B, C, D, H, W)
    xm = torch.randn(B, C, D, H, W)
    mask = torch.ones_like(x)
    model = HiKINet(
        input_shape=(D, H, W),
        fusion_mode="poe",
        missing_refusion_mode="moe",
        kd_loss_type="mse",         # "mse" | "kl" | "cosine"
        seg_activation="sigmoid",   # "none" if you use *WithLogits* losses
        kd_weight=0.1,              # example
        kd_temp=2.0,                # used only when kd_loss_type="kl"
    )
    out = model(x, xm, mask)
    outs = [o.shape if torch.is_tensor(o) else o for o in out[:-2]]
    print("OK. seg_f:", outs[1], "seg_m:", outs[3])
