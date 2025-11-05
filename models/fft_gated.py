
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class MultiHeadSpectralAttention(nn.Module):
    def __init__(self, embed_dim, seq_len, num_heads=4, dropout=0.1, adaptive=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.adaptive = adaptive
        self.freq_bins = self.head_dim // 2 + 1

        self.base_filter = nn.Parameter(torch.ones(num_heads, self.freq_bins))
        self.base_bias = nn.Parameter(torch.full((num_heads, self.freq_bins), -0.1))

        if adaptive:
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.freq_bins * 2)
            )
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)

    def complex_activation(self, x):
        return torch.complex(F.gelu(x.real), F.gelu(x.imag))

    def forward(self, x, return_attention=False):
        B, N, D = x.shape
        x = self.pre_norm(x)
        x_split_heads = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        F_fft = torch.fft.rfft(x_split_heads, dim=-1, norm='ortho')

        if self.adaptive:
            context = x.mean(dim=1)
            adapt_params = self.adaptive_mlp(context).view(B, self.num_heads, self.freq_bins, 2)
            adaptive_scale = adapt_params[..., 0]
            adaptive_bias = adapt_params[..., 1]
        else:
            adaptive_scale = torch.zeros_like(self.base_filter).unsqueeze(0)
            adaptive_bias = torch.zeros_like(self.base_bias).unsqueeze(0)

        effective_filter = (self.base_filter[None, :, None, :] * (1 + adaptive_scale[:, :, None, :])).to(F_fft.dtype)
        effective_bias = (self.base_bias[None, :, None, :] + adaptive_bias[:, :, None, :]).to(F_fft.dtype)
        F_fft_mod = F_fft * effective_filter + effective_bias
        F_fft_nl = self.complex_activation(F_fft_mod)
        x_filtered_split_heads = torch.fft.irfft(F_fft_nl, n=self.head_dim, dim=-1, norm='ortho')
        x_filtered = x_filtered_split_heads.permute(0, 2, 1, 3).contiguous().view(B, N, D)

        if return_attention:
            att_score = x_filtered[:, 1:, :].norm(dim=-1)
            return x_filtered, att_score
        return x_filtered

class FFTTransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1, attention_module=None, drop_path=0.0):
        super().__init__()
        self.attention = attention_module
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbedWithPhysics(nn.Module):
    def __init__(self, patch_size=56, in_chans=3, embed_dim=96, physics_dim=6):
        super().__init__()
        if embed_dim % in_chans != 0:
            raise ValueError("embed_dim must be divisible by in_chans")

        self.patch_embed_r = nn.Conv2d(1, embed_dim // in_chans, kernel_size=patch_size, stride=patch_size)
        self.patch_embed_g = nn.Conv2d(1, embed_dim // in_chans, kernel_size=patch_size, stride=patch_size)
        self.patch_embed_b = nn.Conv2d(1, embed_dim // in_chans, kernel_size=patch_size, stride=patch_size)
        self.ln = nn.LayerNorm(embed_dim)
        self.physics_projection = nn.Linear(physics_dim, embed_dim)
        self.gate_linear = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, patch_feats):
        x_r = self.patch_embed_r(x[:, 0:1, :, :]).flatten(2).transpose(1, 2)
        x_g = self.patch_embed_g(x[:, 1:2, :, :]).flatten(2).transpose(1, 2)
        x_b = self.patch_embed_b(x[:, 2:3, :, :]).flatten(2).transpose(1, 2)

        x_img = torch.cat([x_r, x_g, x_b], dim=-1)
        x_img = self.ln(x_img)
        x_physics_projected = self.physics_projection(patch_feats)
        gate_in = torch.cat([x_img, x_physics_projected], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_in))
        fused = gate * x_img + (1 - gate) * x_physics_projected
        return fused

class FFTPermeabilityPredictorPatchPhysics(nn.Module):
    def __init__(self, patch_size=56, embed_dim=96, num_heads=8, mlp_ratio=4.0, depth=12, num_classes=1):
        super().__init__()
        image_size = 224
        num_patches = (image_size // patch_size) ** 2
        self.seq_len = num_patches + 1

        self.patch_embed = PatchEmbedWithPhysics(
            patch_size=patch_size, in_chans=3, embed_dim=embed_dim, physics_dim=6
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        self.blocks = nn.ModuleList([
            FFTTransformerEncoderBlock(
                embed_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                attention_module=MultiHeadSpectralAttention(
                    embed_dim=embed_dim,
                    seq_len=self.seq_len,
                    num_heads=num_heads
                )
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, patch_feats):
        feat = self.patch_embed(x, patch_feats)
        cls_token = self.cls_token.expand(feat.size(0), -1, -1)
        x = torch.cat((cls_token, feat), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls = x[:, 0]
        return self.head(cls).squeeze()
