import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from .simple_vit import Transformer, pair, posemb_sincos_2d, Attention, FeedForward
from torch.nn.init import xavier_uniform_
import torch.nn as nn

class SimpleViT_v2(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, probe=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.encoder_norm = nn.LayerNorm(dim)

        self.to_latent = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh()
        )
        self.linear_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=1e-6)

    def _linear_init(self, module):
        fan_in = module.weight.shape[1]
        stdev = torch.sqrt(torch.tensor(1/fan_in))/.87962566103423978
        nn.init.trunc_normal_(module.weight, mean=0.0, std=1.0*stdev, a=- 2.0*stdev, b=2.0*stdev)
        nn.init.zeros_(module.bias)

    def _reset_parameters(self):
        self.to_latent[0].apply(self._linear_init)
        self.linear_head[0].apply(self._linear_init)
        self.to_patch_embedding[1].apply(self._linear_init)
        # reset attention
        # reset mlp blocks
        for module in self.modules():
            if isinstance(module, FeedForward) or isinstance(module, Attention):
                module._reset_parameters()



    def forward(self, img, feature_noise={}, get_features=False, get_linear_probes=False, freeze_layers=None):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        if get_features:
            x, features = self.transformer(x, feature_noise=feature_noise, get_features=get_features, freeze_layers=freeze_layers)
        else:
            x = self.transformer(x, feature_noise=feature_noise, freeze_layers=freeze_layers)
        x = self.encoder_norm(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
    def freeze_layers(self, layers_n):
        # freeze all layers prior to layers_n
        for layer in self.transformer.layers[:layers_n]:
            for para in layer.parameters():
                para.requires_grad = False
    def unfreeze_layers(self, layers_n):
        # unfreeze all layers prior to layers_n
        for layer in self.transformer.layers[:layers_n]:
            for para in layer.parameters():
                para.requires_grad = True

if __name__ == "__main__":
    model = SimpleViT_v2(image_size = 224,
            patch_size = 16,
            num_classes = 1000,
            dim = 384,
            depth = 12,
            heads = 6,
            mlp_dim = 768)
    images = torch.randn((32, 3, 224, 224))
    out = model(images)
    import pdb;pdb.set_trace()
    print("out.shape", out.shape)