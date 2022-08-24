import torch
from torch import nn


from models.simple_vit_v2 import SimpleViT_v2
import torch.nn as nn
from models.simple_vit_decoupled import LayerNormDecoupled
# from models.simple_vit_v3 import Transformer_v2
from models.simple_vit import Transformer
from fastargs.decorators import param
class SimpleViTDecoupledLayernorm_v2(SimpleViT_v2):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, probe=False):
        super(SimpleViTDecoupledLayernorm_v2, self).__init__(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = channels, dim_head = dim_head, probe=probe)
        def apply_decoupled_layernorm(mod: nn.Module):
            if not isinstance(mod, LayerNormDecoupled):
                for (name, child) in mod.named_children():
                    if isinstance(child, nn.LayerNorm): 
                        print(f"setting: {mod} {child} to decoupled layernorm")
                        setattr(mod, name, LayerNormDecoupled(child.normalized_shape))
                    else: apply_decoupled_layernorm(child)
        apply_decoupled_layernorm(self)
    
    def make_clean(self):
        for module in self.modules():
            if isinstance(module, LayerNormDecoupled):
                module.make_clean()
    def make_adv(self):
        for module in self.modules():
            if isinstance(module, LayerNormDecoupled):
                module.make_adv()

class Transformer_Universal(Transformer):
    
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, layers=None, step_size=None, radius=None):
        super(Transformer_Universal, self).__init__(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)
        self.universal_feature_noise = {int(l): None for l in layers.split(",")}
        self.step_size = step_size
        self.radius = radius
        self.clean = True
    def forward(self, x, feature_noise = {}, get_features=False, freeze_layers=None):
        for li, (attn, ff) in enumerate(self.layers):
            if li in self.universal_feature_noise and self.training and self.clean == False:
                if self.universal_feature_noise[li] is None:
                    uni_noise = torch.zeros_like(x)
                    uni_noise.requires_grad_(True)
                    self.universal_feature_noise[li] = uni_noise
                else:
                    uni_noise = self.universal_feature_noise[li]
                    if uni_noise.grad is not None:
                        uni_noise.data += self.step_size * torch.sign(uni_noise.grad)
                        uni_noise.data.clamp_(-self.radius, self.radius)
                        uni_noise.grad = None
                x += uni_noise

            x = attn(x) + x
            x = ff(x) + x
        return x
    def make_clean(self):
        self.clean = True
    def make_adv(self):
        self.clean = False

class SimpleViTDecoupled_Universal(SimpleViTDecoupledLayernorm_v2):
    @param('universal_feature_adv.step_size')
    @param('universal_feature_adv.radius')
    @param('universal_feature_adv.layers')
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, probe=False, 
    layers='0', step_size=0.001, radius=0.01):
        super(SimpleViTDecoupled_Universal, self).__init__(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = channels, dim_head = dim_head, probe=probe)
        self.transformer = Transformer_Universal(dim=dim, depth=depth, heads=heads,  dim_head=dim_head, mlp_dim=mlp_dim, layers=layers,
        step_size=step_size, radius=radius)
        def apply_decoupled_layernorm(mod: nn.Module):
            if not isinstance(mod, LayerNormDecoupled):
                for (name, child) in mod.named_children():
                    if isinstance(child, nn.LayerNorm): 
                        print(f"setting: {mod} {child} to decoupled layernorm")
                        setattr(mod, name, LayerNormDecoupled(child.normalized_shape))
                    else: apply_decoupled_layernorm(child)
        apply_decoupled_layernorm(self)
    def make_clean(self):
        for module in self.modules():
            if isinstance(module, LayerNormDecoupled):
                module.make_clean()
            if isinstance(module, Transformer_Universal):
                module.make_clean()
    def make_adv(self):
        for module in self.modules():
            if isinstance(module, LayerNormDecoupled):
                module.make_adv()  
            if isinstance(module, Transformer_Universal):
                module.make_adv()

if __name__ == "__main__":
    # model = SimpleViTDecoupledLayernorm_v2(image_size = 224,
    #         patch_size = 16,
    #         num_classes = 1000,
    #         dim = 384,
    #         depth = 12,
    #         heads = 6,
    #         mlp_dim = 768)
    # images = torch.randn((32, 3, 224, 224))
    # out = model(images)
    # import pdb;pdb.set_trace()

    model = SimpleViTDecoupled_Universal(
            image_size = 224,
            patch_size = 16,
            num_classes = 1000,
            dim = 384,
            depth = 12,
            heads = 6,
            mlp_dim = 384*4, layer_with_universal="0")
    img = torch.randn((32, 3, 224, 224))
    out = model(img)
    model.make_adv()
    out_adv = model(img)
    out_adv.sum().backward()
    out_adv_v2 = model(img)
    model.make_clean()
    out_clean = model(img)
    import pdb;pdb.set_trace()
    pass
    