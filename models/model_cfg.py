import torch.nn as nn
from .model import PTET, PTET_for_test

class net(PTET):
    def __init__(self, **kwargs):
        super(net, self).__init__(
            embed_dims=[16, 32, 64, 128], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, depths=[1, 1, 1, 1], sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
        
class net_for_test(PTET_for_test):
    def __init__(self, **kwargs):
        super(net_for_test, self).__init__(
            embed_dims=[16, 32, 64, 128], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, depths=[1, 1, 1, 1], sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)