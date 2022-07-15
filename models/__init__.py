
from models.mvit_decoupled import MViTDecoupled
from .simple_vit import SimpleViT, SimpleViTTwoHead
from .simple_vit_decoupled import SimpleViTDecoupledLN
from .mvit import MViT
from .mvit_config import get_cfg

def get_arch(arch_name, num_classes, probe=False, split_layer=None):
    if arch_name == 'vit_t':
        return SimpleViT(
            image_size = 256,
            patch_size = 32,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            probe=probe
        )
    elif arch_name == 'vit_s':
        return SimpleViT(
            image_size = 224,
            patch_size = 16,
            num_classes = num_classes,
            dim = 384,
            depth = 12,
            heads = 6,
            mlp_dim = 768,
            probe=probe
        )
    elif arch_name == 'vit_s_twohead':
        return SimpleViTTwoHead(
            image_size = 224,
            patch_size = 16,
            num_classes = num_classes,
            dim = 384,
            depth = 12,
            heads = 6,
            mlp_dim = 768,
            probe=probe,
            split_layer=split_layer
        )
    elif arch_name == 'vit_s_decoupled':
        return SimpleViTDecoupledLN(
            image_size = 224,
            patch_size = 16,
            num_classes = num_classes,
            dim = 384,
            depth = 12,
            heads = 6,
            mlp_dim = 768,
            probe=probe
        )
    elif arch_name == 'vit_m':
        return SimpleViT(
            image_size = 224,
            patch_size = 16,
            num_classes = num_classes,
            dim = 512,
            depth = 12,
            heads = 8,
            mlp_dim = 2048,
            probe=probe
        )
    elif arch_name == 'vit_b':
        return SimpleViT(
            image_size = 224,
            patch_size = 16,
            num_classes = num_classes,
            dim = 768,
            depth = 12,
            heads = 12,
            mlp_dim = 3072,
            probe=probe
        )
    if arch_name == 'mvit':
        cfg = get_cfg()
        cfg.merge_from_list(
            ['MVIT.DROPPATH_RATE', '0.1',
            'MVIT.DEPTH', '10',
            'MVIT.DIM_MUL', '[[1, 2.0], [3, 2.0], [8, 2.0]]',
            'MVIT.HEAD_MUL', '[[1, 2.0], [3, 2.0], [8, 2.0]]',
            'MVIT.POOL_KVQ_KERNEL', '[3, 3]',
            'MVIT.POOL_KV_STRIDE_ADAPTIVE', '[4, 4]',
            'MVIT.POOL_Q_STRIDE', '[[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 2, 2], [9, 1, 1]]',
            'MODEL.NUM_CLASSES', num_classes
            ]
        )
        return MViT(cfg,
            probe=probe)
    elif arch_name == 'mvit_decoupled':
        cfg = get_cfg()
        cfg.merge_from_list(
            ['MVIT.DROPPATH_RATE', '0.1',
            'MVIT.DEPTH', '10',
            'MVIT.DIM_MUL', '[[1, 2.0], [3, 2.0], [8, 2.0]]',
            'MVIT.HEAD_MUL', '[[1, 2.0], [3, 2.0], [8, 2.0]]',
            'MVIT.POOL_KVQ_KERNEL', '[3, 3]',
            'MVIT.POOL_KV_STRIDE_ADAPTIVE', '[4, 4]',
            'MVIT.POOL_Q_STRIDE', '[[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 2, 2], [9, 1, 1]]',
            'MODEL.NUM_CLASSES', num_classes
            ]
        )
        return MViTDecoupled(cfg,
            probe=probe)
    
    elif arch_name == 'mvit_s':
        cfg = get_cfg()
        cfg.merge_from_list(
            ['MVIT.DROPPATH_RATE', '0.1',
            'MVIT.DEPTH', '16',
            'MVIT.DIM_MUL', '[[1, 2.0], [3, 2.0], [14, 2.0]]',
            'MVIT.HEAD_MUL', '[[1, 2.0], [3, 2.0], [14, 2.0]]',
            'MVIT.POOL_KVQ_KERNEL', '[3, 3]',
            'MVIT.POOL_KV_STRIDE_ADAPTIVE', '[4, 4]',
            'MVIT.POOL_Q_STRIDE', '[[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 1, 1], [9, 1, 1], [10, 1, 1], [11, 1, 1], [12, 1, 1], [13, 1, 1], [14, 2, 2], [15, 1, 1]]',
            'MODEL.NUM_CLASSES', num_classes
            ]
        )
        return MViT(cfg,
            probe=probe)
    elif arch_name == 'mvit_s_decoupled':
        cfg = get_cfg()
        cfg.merge_from_list(
            ['MVIT.DROPPATH_RATE', '0.1',
            'MVIT.DEPTH', '16',
            'MVIT.DIM_MUL', '[[1, 2.0], [3, 2.0], [14, 2.0]]',
            'MVIT.HEAD_MUL', '[[1, 2.0], [3, 2.0], [14, 2.0]]',
            'MVIT.POOL_KVQ_KERNEL', '[3, 3]',
            'MVIT.POOL_KV_STRIDE_ADAPTIVE', '[4, 4]',
            'MVIT.POOL_Q_STRIDE', '[[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 1, 1], [9, 1, 1], [10, 1, 1], [11, 1, 1], [12, 1, 1], [13, 1, 1], [14, 2, 2], [15, 1, 1]]',
            'MODEL.NUM_CLASSES', num_classes
            ]
        )
        return MViTDecoupled(cfg,
            probe=probe)

    elif arch_name == 'mvit_b':
        cfg = get_cfg()
        cfg.merge_from_list(
            ['MVIT.DROPPATH_RATE', '0.3',
            'MVIT.DEPTH', '24',
            'MVIT.DIM_MUL', '[[2, 2.0], [5, 2.0], [21, 2.0]]',
            'MVIT.HEAD_MUL', '[[2, 2.0], [5, 2.0], [21, 2.0]]',
            'MVIT.POOL_KVQ_KERNEL', '[3, 3]',
            'MVIT.POOL_KV_STRIDE_ADAPTIVE', '[4, 4]',
            'MVIT.POOL_Q_STRIDE', '[[0, 1, 1], [1, 1, 1], [2, 2, 2], [3, 1, 1], [4, 1, 1], [5, 2, 2], [6, 1, 1], [7, 1, 1], [8, 1, 1], [9, 1, 1], [10, 1, 1], [11, 1, 1], [12, 1, 1], [13, 1, 1], [14, 1, 1], [15, 1, 1], [16, 1, 1], [17, 1, 1], [18, 1, 1], [19, 1, 1], [20, 1, 1], [21, 2, 2], [22, 1, 1], [23, 1, 1]]',
            'MODEL.NUM_CLASSES', num_classes
            ]
        )
        return MViT(cfg,
            probe=probe)
    