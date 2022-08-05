import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class LayerNormDecoupled(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layernorm_clean = nn.LayerNorm(dim)
        self.layernorm_adv = nn.LayerNorm(dim)
        self.clean = True
    def forward(self, x):
        if self.clean:
            return self.layernorm_clean(x)
        else:
            return self.layernorm_adv(x)
    def make_clean(self):
        self.clean = True
    def make_adv(self):
        self.clean = False
    def reset_layernorm(self):
        self.layernorm_adv.weight.data = self.layernorm_clean.weight.data.clone()
        self.layernorm_adv.bias.data = self.layernorm_clean.bias.data.clone()
        

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            LayerNormDecoupled(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = LayerNormDecoupled(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, feature_noise = {}, get_features=False, freeze_layers=None):
        if get_features:
            features = {}
        for li, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            if li in feature_noise:
                if feature_noise[li] is None:
                    feature_noise[li] = torch.zeros_like(x, requires_grad=True)
                x += feature_noise[li]
            if freeze_layers is not None and li == freeze_layers - 1:
                x = x.detach()
            if get_features:
                features[li] = x
        if get_features:
            return x, features
        else:
            return x

class TransformerDecoupledLN(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, feature_noise = {}, get_features=False, freeze_layers=None):
        if get_features:
            features = {}
        for li, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            if li in feature_noise:
                if feature_noise[li] is None:
                    feature_noise[li] = torch.zeros_like(x, requires_grad=True)
                x += feature_noise[li]
            if freeze_layers is not None and li == freeze_layers - 1:
                x = x.detach()
            if get_features:
                features[li] = x
        if get_features:
            return x, features
        else:
            return x





class SimpleViTDecoupledLN(nn.Module):
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

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            LayerNormDecoupled(dim),
            nn.Linear(dim, num_classes)
        )
        if probe:
            self.linear_probes = nn.ModuleList()
            for d in range(depth):
                self.linear_probes.append(nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, num_classes)
                ))

    def forward(self, img, feature_noise={}, get_features=False, get_linear_probes=False, freeze_layers=None):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        if get_features:
            x, features = self.transformer(x, feature_noise=feature_noise, get_features=get_features, freeze_layers=freeze_layers)
        else:
            x = self.transformer(x, feature_noise=feature_noise, freeze_layers=freeze_layers)
        if get_linear_probes:
            for k, v in features.items():
                features[k] = self.linear_probes[k](v.mean(dim=1).detach())
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        if get_features:
            return self.linear_head(x), features
        else:
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
    def make_clean(self):
        for module in self.modules():
            if isinstance(module, LayerNormDecoupled):
                module.make_clean()
    def make_adv(self):
        for module in self.modules():
            if isinstance(module, LayerNormDecoupled):
                module.make_adv()
    def reset_layernorm(self):
        for module in self.modules():
            if isinstance(module, LayerNormDecoupled):
                module.reset_layernorm()
    
        
                

