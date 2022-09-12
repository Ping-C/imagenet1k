
# forward with jax
# forward with torch
# given a single input
# from .simple_vit_v3 import SimpleViT_v3
import sys
sys.path.append('/cmlscratch/pchiang/big_vision')
from big_vision.models.vit import Model
from .simple_vit_v3 import SimpleViT_v3, posemb_sincos_2d
import torch
import jax
from functools import partial
import jax.numpy as jnp
import flax
import numpy as np
from einops import rearrange
from collections import defaultdict
import matplotlib.pyplot as plt




rng = jax.random.PRNGKey(0)
jax_model = Model(1000, variant='S/16', rep_size=True,
      pool_type='gap',
      posemb='sincos2d')
@partial(jax.jit, backend="cpu")
def init(rng):
    shape = (224, 224, 3)
    dummy_input = jnp.zeros((1,) + shape, jnp.float32)
    params = flax.core.unfreeze(jax_model.init(rng, dummy_input))["params"]

    return params

params = init(rng)


image = jax.random.normal(rng, (1, 224, 224, 3))
jax_logits, jax_inter_outputs = jax_model.apply({"params": params}, image)




# compare two embedding
def copy_embedding_linear(block_jax, block_torch):
    block_torch.weight.data = torch.from_numpy(np.array(block_jax['kernel'])).reshape(-1, 384).permute(1, 0)
    block_torch.bias.data = torch.from_numpy(np.array(block_jax['bias']))


def copy_single_block(block_jax, block_torch):
    # copy layernorm 0
    copy_norm(block_jax['LayerNorm_0'], block_torch[0].norm)
    # copy attention
    copy_attention(block_jax[f'MultiHeadDotProductAttention_0'], block_torch[0])
    # copy layernorm 1
    copy_norm(block_jax['LayerNorm_1'], block_torch[1].net[0])
    # copy fc
    copy_fc(block_jax[f'MlpBlock_0'], block_torch[1])

def copy_norm(norm_jax, norm_torch):
    norm_scale= torch.from_numpy(np.array(norm_jax['scale']))
    norm_bias= torch.from_numpy(np.array(norm_jax['bias']))
    norm_torch.weight.data = norm_scale
    norm_torch.bias.data = norm_bias

def copy_attention(att_jax, att_torch):
    key_weight = torch.from_numpy(np.array(att_jax['key']['kernel']))
    key_bias = torch.from_numpy(np.array(att_jax['key']['bias']))
    value_weight = torch.from_numpy(np.array(att_jax['value']['kernel']))
    value_bias = torch.from_numpy(np.array(att_jax['value']['bias']))
    query_weight = torch.from_numpy(np.array(att_jax['query']['kernel']))
    query_bias = torch.from_numpy(np.array(att_jax['query']['bias']))
    out_weight = torch.from_numpy(np.array(att_jax['out']['kernel']))
    out_bias = torch.from_numpy(np.array(att_jax['out']['bias']))
    qkv_weight = torch.cat([query_weight.reshape(384, -1), key_weight.reshape(384, -1), value_weight.reshape(384, -1)], dim=1)
    qkv_bias = torch.cat([query_bias.view(-1), key_bias.view(-1), value_bias.view(-1)], dim=0)

    att_torch.to_qkv.weight.data = qkv_weight.transpose(0, 1)
    att_torch.to_qkv.bias.data = qkv_bias
    att_torch.to_out.weight.data = out_weight.reshape(-1, 384).permute(1, 0)
    att_torch.to_out.bias.data = out_bias

def copy_fc(fc_jax, fc_torch):
    copy_dense(fc_jax['Dense_0'], fc_torch.net[1])
    copy_dense(fc_jax['Dense_1'], fc_torch.net[3])

def copy_dense(dense_jax, dense_torch):
    w0 = torch.from_numpy(np.array(dense_jax['kernel']))
    b0 = torch.from_numpy(np.array(dense_jax['bias']))
    dense_torch.weight.data = w0.permute(1, 0)
    dense_torch.bias.data = b0

def copy_model(model_jax, model_torch):
    copy_embedding_linear(model_jax['embedding'], model_torch.to_patch_embedding[1])
    for i in range(12):
        copy_single_block(model_jax['Transformer'][f'encoderblock_{i}'], model_torch.transformer.layers[i])
    copy_norm(model_jax['Transformer']['encoder_norm'], model_torch.encoder_norm)
    copy_dense(model_jax['pre_logits'], model_torch.to_latent[0])
    copy_dense(model_jax['head'], model_torch.linear_head[0])
# copy image over
image_torch = torch.from_numpy(np.array(image)).permute(0, 3, 1, 2)

model_torch = SimpleViT_v3(
            image_size = 224,
            patch_size = 16,
            num_classes = 1000,
            dim = 384,
            depth = 12,
            heads = 6,
            mlp_dim = 384*4
        )

copy_model(params, model_torch)



def get_intermediate_outputs(model_torch):
    rec_dd = lambda: defaultdict(rec_dd)
    torch_inter_outputs = rec_dd()
    embed = model_torch.to_patch_embedding(image_torch)

    # embedding with positional encoding
    embed_withpe = rearrange(embed, 'b ... d -> b (...) d') + posemb_sincos_2d(embed)
    torch_inter_outputs['stem'] = embed
    torch_inter_outputs['with_posemb'] = embed_withpe

    x = embed_withpe
    for i in range(12):
        x = model_torch.transformer.layers[i][0](x) + x
        torch_inter_outputs['encoder'][f'block{i:0>2}']['+sa'] = x
        x = model_torch.transformer.layers[i][1](x) + x
        torch_inter_outputs['encoder'][f'block{i:0>2}']['+mlp'] = x

    x = model_torch.encoder_norm(x)
    torch_inter_outputs['encoded'] = x
    x = x.mean(dim=1)
    torch_inter_outputs['head_input'] = x
    x = model_torch.to_latent(x)
    torch_inter_outputs['pre_logits'] = x
    x = model_torch.linear_head(x)
    torch_inter_outputs['logits'] = x
    return torch_inter_outputs

def traverse(outputs_torch, outputs_jax, prefix=""):
    if prefix != '':
        prefix_display = prefix+'.'
    else:
        prefix_display = prefix
    for k, v in outputs_torch.items():
        if torch.is_tensor(v):
            # evaluate differences
            max_diff = (v - torch.from_numpy(np.array(outputs_jax[k]))).abs().max()
            
            print(f"{prefix_display+k:<30} max difference:{max_diff.item()}")
        else:
            traverse(outputs_torch[k], outputs_jax[k], prefix=f"{prefix_display}{k}")

torch_inter_outputs = get_intermediate_outputs(model_torch)
traverse(torch_inter_outputs, jax_inter_outputs)


# compare distribution of the initialization

# reinitialize the weights
model_torch = SimpleViT_v3(
            image_size = 224,
            patch_size = 16,
            num_classes = 1000,
            dim = 384,
            depth = 12,
            heads = 6,
            mlp_dim = 384*4
        )
model_torch._reset_parameters_v3()
weight_dict = {
    'embedding.kernel': model_torch.to_patch_embedding[1].weight,
    'embedding.bias': model_torch.to_patch_embedding[1].bias,
    'Transformer.encoderblock_0.LayerNorm_0.scale': model_torch.transformer.layers[0][0].norm.weight,
    'Transformer.encoderblock_0.LayerNorm_0.bias': model_torch.transformer.layers[0][0].norm.bias,
    'Transformer.encoderblock_0.MultiHeadDotProductAttention_0.key.kernel': model_torch.transformer.layers[0][0].to_qkv.weight[:384],
    'Transformer.encoderblock_0.MultiHeadDotProductAttention_0.key.bias': model_torch.transformer.layers[0][0].to_qkv.bias[:384],
    'Transformer.encoderblock_0.MultiHeadDotProductAttention_0.query.kernel': model_torch.transformer.layers[0][0].to_qkv.weight[:384],
    'Transformer.encoderblock_0.MultiHeadDotProductAttention_0.query.bias': model_torch.transformer.layers[0][0].to_qkv.bias[:384],
    'Transformer.encoderblock_0.MultiHeadDotProductAttention_0.value.kernel': model_torch.transformer.layers[0][0].to_qkv.weight[:384],
    'Transformer.encoderblock_0.MultiHeadDotProductAttention_0.value.bias': model_torch.transformer.layers[0][0].to_qkv.bias[:384],
    'Transformer.encoderblock_0.MultiHeadDotProductAttention_0.out.kernel': model_torch.transformer.layers[0][0].to_out.weight,
    'Transformer.encoderblock_0.MultiHeadDotProductAttention_0.out.bias': model_torch.transformer.layers[0][0].to_out.bias,
    'Transformer.encoderblock_0.LayerNorm_1.scale': model_torch.transformer.layers[0][1].net[0].weight,
    'Transformer.encoderblock_0.LayerNorm_1.bias': model_torch.transformer.layers[0][1].net[0].bias,
    'Transformer.encoderblock_0.MlpBlock_0.Dense_0.kernel': model_torch.transformer.layers[0][1].net[1].weight,
    'Transformer.encoderblock_0.MlpBlock_0.Dense_0.bias': model_torch.transformer.layers[0][1].net[1].bias,
    'Transformer.encoderblock_0.MlpBlock_0.Dense_1.kernel': model_torch.transformer.layers[0][1].net[3].weight,
    'Transformer.encoderblock_0.MlpBlock_0.Dense_1.bias': model_torch.transformer.layers[0][1].net[3].bias,
    'pre_logits.kernel': model_torch.to_latent[0].weight,
    'pre_logits.bias': model_torch.to_latent[0].bias,
    'head.kernel': model_torch.linear_head[0].weight,
    'head.bias': model_torch.linear_head[0].bias,
}


fig, axes = plt.subplots(3, len(weight_dict), figsize=(len(weight_dict)*4, 12))
for i, (k, weight_torch) in enumerate(weight_dict.items()):
    weight_torch = weight_torch.flatten().detach().numpy()
    weight_jax = params
    for key in k.split('.'):
        weight_jax = weight_jax[key]
    weight_jax = np.array(weight_jax).reshape(-1)
    b_u = max(weight_torch.max()*1.2, weight_jax.max()*1.2)
    b_l = -b_u
    if weight_torch.max() == 0:
        bins = np.arange(-0.1, 0.1, 0.1/10)
    elif weight_torch.max() == weight_torch.min():
        bins = np.arange(b_l, b_u, b_u/10)
    else:
        bins = np.arange(b_l, b_u, b_u/100)
    axes[0][i].title.set_text(k.replace('.', '.\n'))
    axes[0][i].hist(weight_torch, color='g', alpha=0.2, bins=bins)
    axes[1][i].hist(weight_jax, color='r', alpha=0.2, bins=bins)
    axes[2][i].hist(weight_torch, color='g', alpha=0.2, bins=bins)
    axes[2][i].hist(weight_jax, color='r', alpha=0.2, bins=bins)
plt.savefig('distribution_comp.png')
import pdb; pdb.set_trace()
pass
