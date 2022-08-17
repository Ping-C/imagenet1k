import torch
from torch import nn


from .simple_vit_v2 import SimpleViT_v2
import torch.nn as nn
from .simple_vit_decoupled import LayerNormDecoupled

class SimpleViTDecoupledLayernorm_v2(SimpleViT_v2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        
if __name__ == "__main__":
    model = SimpleViTDecoupledLayernorm_v2(image_size = 224,
            patch_size = 16,
            num_classes = 1000,
            dim = 384,
            depth = 12,
            heads = 6,
            mlp_dim = 768)
    images = torch.randn((32, 3, 224, 224))
    out = model(images)
    import pdb;pdb.set_trace()