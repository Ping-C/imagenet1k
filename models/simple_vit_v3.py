import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_
import torch.nn as nn

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

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)
    def _reset_parameters(self):
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=1e-6)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = True)
        self.to_out = nn.Linear(inner_dim, dim, bias = True)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

    def _reset_parameters(self):
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, feature_noise = {}, get_features=False, freeze_layers=None, get_attention_maps=False):
        if get_features:
            features = {}
        attmaps = []
        for li, (attn, ff) in enumerate(self.layers):
            attn_x, attmap = attn(x)
            if get_attention_maps:
                attmaps.append(attmap[:, None])
            x = attn_x + x
            x = ff(x) + x
            if li in feature_noise:
                if feature_noise[li] is None:
                    feature_noise[li] = torch.zeros_like(x, requires_grad=True)
                x += feature_noise[li]
            if freeze_layers is not None and li == freeze_layers - 1:
                x = x.detach()
            if get_features:
                features[li] = x
        
        if get_attention_maps:
            attmaps = torch.cat(attmaps, dim=1)
            return x, attmaps
        if get_features:
            return x, features
        else:
            return x


class SimpleViT_v3(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, probe=False):
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
        self._reset_parameters()
    
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
        
        nn.init.zeros_(self.linear_head[0].weight.data)
        for module in self.modules():
            if isinstance(module, Attention):
                fan_in = module.to_qkv.weight.shape[0]
                fan_out = module.to_qkv.weight.shape[1]
                modify_gain = torch.sqrt(torch.tensor((fan_in+fan_out)/(fan_in/3+fan_out)))
                xavier_uniform_(module.to_qkv.weight, gain=modify_gain)
        # fix initialization 
        

    def forward(self, img, feature_noise={}, get_features=False, get_linear_probes=False, freeze_layers=None, get_attention_maps=False):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        if get_attention_maps:
            x, att_maps = self.transformer(x, get_attention_maps=get_attention_maps)
        if get_features:
            x, features = self.transformer(x, feature_noise=feature_noise, get_features=get_features, freeze_layers=freeze_layers)
        else:
            x = self.transformer(x, feature_noise=feature_noise, freeze_layers=freeze_layers)
        x = self.encoder_norm(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        if get_attention_maps:
            return self.linear_head(x), att_maps
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

                

class PyramidGenerator(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, probe=False):
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

        self.to_patch_0 = nn.Sequential(
            nn.Linear(dim, patch_dim),
            nn.Sigmoid(),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, 
                      h=image_height // patch_height, 
                      w=image_width // patch_width),
        )
        self.to_patch_1 = nn.Sequential(
            nn.Linear(dim, patch_dim//(16*16)),
            nn.Sigmoid(),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height//16, p2 = patch_width//16, 
                      h=image_height // patch_height, 
                      w=image_width // patch_height)
        )
        self.to_patch_2 = nn.Sequential(
            nn.Linear(dim, patch_dim//(16*16)),
            nn.Sigmoid(),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height//16, p2 = patch_width//16, 
                      h=image_height // patch_height, 
                      w=image_width // patch_height)
        )
        self._reset_parameters()
    
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
        self.to_patch_0[0].apply(self._linear_init)
        self.to_patch_1[0].apply(self._linear_init)
        self.to_patch_2[0].apply(self._linear_init)
        self.to_patch_embedding[1].apply(self._linear_init)

        # reset attention
        # reset mlp blocks
        for module in self.modules():
            if isinstance(module, FeedForward) or isinstance(module, Attention):
                module._reset_parameters()
        
        for module in self.modules():
            if isinstance(module, Attention):
                fan_in = module.to_qkv.weight.shape[0]
                fan_out = module.to_qkv.weight.shape[1]
                modify_gain = torch.sqrt(torch.tensor((fan_in+fan_out)/(fan_in/3+fan_out)))
                xavier_uniform_(module.to_qkv.weight, gain=modify_gain)
        

    def forward(self, img, feature_noise={}, get_features=False, get_linear_probes=False, freeze_layers=None, get_attention_maps=False):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        x = self.transformer(x, feature_noise=feature_noise, freeze_layers=freeze_layers)
        x = self.encoder_norm(x)
        x0 = self.to_patch_0(x)
        x0 += torch.nn.functional.interpolate(self.to_patch_1(x), x0.shape[2:])*10
        x_32 = torch.nn.functional.interpolate(self.to_patch_2(x), size=(7,7))
        x0 += torch.nn.functional.interpolate(x_32, x0.shape[2:])*20
        
        return x0
    def flip_grad(self, factor=1):
        for para in self.parameters():
            para.grad *= -1 * factor
if __name__ == "__main__":
    import pdb;pdb.set_trace()
    generator = PyramidGenerator(image_size=224, 
                                 patch_size=16, 
                                 dim=384, 
                                 depth=1, 
                                 heads=6, 
                                 mlp_dim=384*4)
    images = torch.randn((5, 3, 224, 224))
    out = generator(images)
    pass
    