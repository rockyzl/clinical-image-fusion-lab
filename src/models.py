"""Frozen image features and five explicitly scoped trainable prediction heads."""
import torch
from torch import nn
from vendor.healnet.healnet import HealNet
from vendor.metablock.metablock import MetaBlock


class FusionModel(nn.Module):
    def __init__(self, name, tab_dim, config):
        super().__init__()
        self.name = name
        if name == "clinical_linear":
            self.head = nn.Linear(tab_dim, 6)
        elif name == "image_linear":
            self.head = nn.Linear(512, 6)
        elif name == "concat_linear":
            self.head = nn.Linear(512 + tab_dim, 6)
        elif name == "metablock":
            channels = config["architectures"]["metablock"]["image_projection"]
            self.projection = nn.Linear(512, channels)
            self.fusion = MetaBlock(channels, tab_dim)
            self.head = nn.Linear(channels, 6)
        elif name == "healnet":
            a = config["architectures"]["healnet"]
            self.projection = nn.Linear(512, a["image_projection"])
            keys = ["depth", "l_c", "l_d", "x_heads", "l_heads", "cross_dim_head", "latent_dim_head", "num_freq_bands", "max_freq", "self_per_cross_attn", "ff_dropout", "attn_dropout", "snn", "weight_tie_layers", "fourier_encode_data", "final_classifier_head"]
            self.fusion = HealNet(n_modalities=2, channel_dims=[tab_dim, a["image_projection"]], num_spatial_axes=[1, 2], out_dims=6, **{k: a[k] for k in keys})
        else:
            raise ValueError(f"Unknown model: {name}")

    def forward(self, images, clinical):
        # images contain spatial feature maps from a FROZEN ImageNet encoder.
        if self.name == "clinical_linear":
            return self.head(clinical)
        if self.name == "image_linear":
            return self.head(images.mean((1, 2)))
        if self.name == "concat_linear":
            return self.head(torch.cat([images.mean((1, 2)), clinical], dim=1))
        image_tokens = self.projection(images)
        if self.name == "metablock":
            gated = self.fusion(image_tokens.flatten(1, 2).transpose(1, 2), clinical)
            return self.head(gated.mean(-1))
        return self.fusion([clinical.unsqueeze(1), image_tokens])


class RawFeatureWrapper(nn.Module):
    """Same preprocessing at local/API/browser inference as the saved experiment."""
    def __init__(self, model, image_mean, image_scale):
        super().__init__()
        self.model = model
        self.register_buffer("image_mean", torch.as_tensor(image_mean, dtype=torch.float32))
        self.register_buffer("image_scale", torch.as_tensor(image_scale, dtype=torch.float32))

    def forward(self, image_features, clinical):
        return self.model((image_features - self.image_mean) / self.image_scale, clinical)
