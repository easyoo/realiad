import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from timm.models.resnet import Bottleneck

from model import get_model
from model import MODEL

import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import numpy as np
from hilbert import decode, encode
from pyzorder import ZOrderIndexer


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)*x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)*x0

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=1,dilation=1):
        super().__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,padding=padding,stride=stride, groups=in_channels,dilation=1),
            nn.InstanceNorm2d(in_channels),nn.SiLU())
        self.pconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),nn.SiLU())
 
    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)



def add_jitter(feature_tokens, scale=20, prob=1):
    """
    Args:
        feature_tokens : B N C
    """
    import random
    device = feature_tokens.device
    if random.uniform(0, 1) <= prob:
        B, N, C = feature_tokens.shape
        feature_norms = (feature_tokens.norm(dim=2).unsqueeze(2) / C)
        jitter = torch.randn((B, N, C)).to(device)
        jitter = jitter * feature_norms * scale
        feature_tokens = feature_tokens + jitter
    return feature_tokens

class Mlp(nn.Module):
    def __init__(self, 
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.SiLU,
                 drop=0.):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Attention(nn.Module):
    def __init__(self,q,k,v,norm,act,pe=None):
        super().__init__()
        self.q=q
        self.k=k
        self.v=v
        self.scale = self.q.size(-1)**-0.5
        b,h,n,c = self.q.shape
        
        self.ffn = Mlp(h*c)
    
        self.pe = pe
    def forward(self):
        # qkv shape b h N c ::pe b h N n
        b,h,n,c = self.q.shape
        self.k = self.k.transpose(-1,-2)# b h c n
        attn_score = self.pe + ((self.q @ self.k)*self.scale).softmax(-1) # b h N n
        atten = attn_score @ self.v # b h N c
        out = atten.permute(0,2,1,3).reshape(b,n,-1) # b N h*c
        out = self.ffn(out) # b N h*c        
        return out.contiguous() # b N h*c

class ToProxy(nn.Module):
    def __init__(self,dim=512,head_num = 8,agent_q_num=7**2,agent_kv_num=7**2,norm=nn.InstanceNorm2d,act=nn.SiLU):
        super().__init__()
        self.agent_q_num = agent_q_num
        self.learnable_q = nn.Parameter(torch.randn(1,agent_q_num,dim))
        self.learnable_kv = nn.Parameter(torch.randn(1,agent_kv_num,dim))
        self.q = nn.Linear(dim,dim)
        self.kv_x = nn.Linear(dim,dim*2)
        self.kv_y = nn.Linear(dim,dim*2)
        self.norm = norm(dim)
        self.act = act()
        self.head = head_num
        self.q_2 = nn.Linear(dim,dim)
    def forward(self,x):
        b,h,w,c = x.shape
        head = self.head
        x = x.permute(0,3,1,2).reshape(b,c,-1).transpose(1,2) # bnc
        learnable_q = self.learnable_q.repeat(b,1,1) # b N(agent_q_num) c
        q = self.q(learnable_q).reshape(b,self.agent_q_num,head,c//head).permute(0,2,1,3) # b h N c
        k,v = self.kv_x(x).chunk(2,dim=-1)
        k = k.reshape(b,h*w,head,c//head).permute(0,2,1,3) # b h n c
        v = v.reshape(b,h*w,head,c//head).permute(0,2,1,3) # b h n c
        atn = Attention(q,k,v,self.norm,self.act).to(x.device) 
        front_out = atn() # b N C
        # 第二阶段
        q = self.q_2(front_out).reshape(b,self.agent_q_num,head,c//head).permute(0,2,1,3) # b h N c
        learnable_v = self.learnable_kv.repeat(b,1,1) # b N1(agent_kv_num) c
        b,N1,c = learnable_v.shape
        k,v = self.kv_y(learnable_v).chunk(2,dim=-1)
        k = k.reshape(b,N1,head,c//head).permute(0,2,1,3) # b h n c
        v = v.reshape(b,N1,head,c//head).permute(0,2,1,3) # b h n c
        atn = Attention(q,k,v,self.norm,self.act).to(x.device)
        out = atn() # b N C
        # 上采样到原始尺寸
        h1 = w1 = int(self.agent_q_num**0.5)
        out = out.reshape(b,h1,w1,-1).permute(0,3,1,2) # b c h1 w1
        out = F.interpolate(out,(h,w)).permute(0,2,3,1).contiguous()
        return out
        
class ToProxy2(nn.Module):
    def __init__(self,dim=512,head_num = 8,agent_q_num=8**2,agent_kv_num=8**2,norm=nn.InstanceNorm2d,act=nn.SiLU):
        super().__init__()
        self.agent_q_num = agent_q_num
        self.learnable_q = nn.Parameter(torch.zeros(1,agent_q_num,dim))
        self.learnable_kv = nn.Parameter(torch.zeros(1,agent_kv_num,dim))
        self.q = nn.Linear(dim,dim)
        self.kv_x = nn.Linear(dim,dim*2)
        self.kv_y = nn.Linear(dim,dim*2)
        self.norm = norm(dim)
        self.act = act()
        self.head = head_num
        self.q_2 = nn.Linear(dim,dim)
        self.pe1 = nn.Parameter(torch.zeros(1,head_num,1,agent_kv_num))
        self.pe2 = nn.Parameter(torch.zeros(1,head_num,agent_q_num,1))
        
    def forward(self,x):
        b,h,w,c = x.shape
        head = self.head
         
         
         
        learnable_q = self.learnable_q.repeat(b,1,1) # b N(agent_q_num) c
        q = self.q(learnable_q).reshape(b,self.agent_q_num,head,c//head).permute(0,2,1,3) # b h N c
        k,v = self.kv_x(x).chunk(2,dim=-1) # b n c 
        
        k = k.reshape(b,h*w,head,c//head).permute(0,2,1,3) # b h n c
        v = v.reshape(b,h*w,head,c//head).permute(0,2,1,3) # b h n c
        pe2 = self.pe2.repeat(b,1,1,h*w)
        atn = Attention(q,k,v,self.norm,self.act,pe=pe2).to(x.device) 
        fin_out = atn() # b agent_q_num C
        # 上采样到原始尺寸
        h1 = w1 = int(self.agent_q_num**0.5)
        out = fin_out.reshape(b,h1,w1,-1).permute(0,3,1,2) # b c h1 w1
        out = F.interpolate(out,(h,w)).permute(0,2,3,1) #b h w c 
        out = out + x
        mskip = out
         
        # 
        q = self.q_2(out).reshape(b,h*w,-1).reshape(b,h*w,head,c//head).permute(0,2,1,3) # b h N c
        learnable_v = self.learnable_kv.repeat(b,1,1) # b N1(agent_kv_num) c
        pe1 = self.pe1.repeat(b,1,h*w,1)
        b,N1,c = learnable_v.shape
        k,v = self.kv_y(learnable_v).chunk(2,dim=-1)
        k = k.reshape(b,N1,head,c//head).permute(0,2,1,3) # b h n c
        v = v.reshape(b,N1,head,c//head).permute(0,2,1,3) # b h n c
        atn = Attention(q,k,v,self.norm,self.act,pe=pe1).to(x.device)
        out = atn().reshape(b,h,w,-1)+mskip+x
        return out.contiguous()

class ProxyAttentionBlock(nn.Module):
    def __init__(self,agent_num=16,dim = 512,head_num = 8,dp=0.,attn_dp=0.):
        super().__init__()
        self.agent_tokens_v = nn.Parameter(torch.randn(1,agent_num,dim))
        self.agent_tokens_k = nn.Parameter(torch.randn(1,agent_num,dim))
        self.head_num = head_num
        self.agent_num = agent_num
        self.k = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)
        self.q = nn.Linear(dim,dim)
        self.scale = (dim//head_num) ** -0.5
        self.proj = nn.Linear(dim,dim)
        self.dp = nn.Dropout(dp)
        self.atndp = nn.Dropout(attn_dp)
        # self.proxyX = ProxyX(agent_num=agent_num,agent=self.agent_tokens,dim=dim,head_num=head_num,dp=dp,attn_dp=attn_dp)
        
        # self.norm = nn.InstanceNorm2d(dim)
        self.cat = ChannelAttention(agent_num)
    def forward(self,x):
        b,h,w,c = x.shape
        # x1 = self.proxyX(x)
        x1 = x
        x1 = x1.reshape(b,-1,c) # b hw c
        q = self.q(x1).reshape(b,h*w,self.head_num,c//self.head_num).permute(0,2,1,3) # b h hw c
        agent_tokens_k = self.agent_tokens_k.repeat(b,1,1) # b n c
        agent_tokens_v = self.agent_tokens_v.repeat(b,1,1) # b n c
        k,v = self.k(agent_tokens_k),self.v(agent_tokens_v)
        k = k.reshape(b,self.agent_num,self.head_num,c//self.head_num).permute(0,2,3,1) #b h c n
        attn_score = ((q @ k) * self.scale).softmax(-1) # b h N n
        attn_score = self.atndp(attn_score)
        attn_score = self.cat(attn_score.permute(0,3,1,2)).permute(0,2,3,1)
        v = v.reshape(b,self.agent_num,self.head_num,c//self.head_num).permute(0,2,1,3) #b h n c
        atte = (attn_score @ v).permute(0,2,1,3).reshape(b,h,w,-1) # b h w c
        out = self.dp(self.proj(atte)).contiguous()
        return out

class ProxyX(nn.Module):
    def __init__(self,agent_num=32,agent=None,dim = 512,head_num = 8,dp=0.,attn_dp=0.):
        super().__init__()
        self.agent_tokens = agent
        self.head_num = head_num
        self.agent_num = agent_num
        self.kv = nn.Linear(dim,dim*2)
        self.q = nn.Linear(dim,dim)
        self.scale = (dim//head_num) ** -0.5
        self.proj = nn.Linear(dim,dim)
        self.dp = nn.Dropout(dp)
        self.atndp = nn.Dropout(attn_dp)
        
    def forward(self,x):
        b,h,w,c = x.shape
        x1 = x.reshape(b,-1,c) # b hw c
        q = self.q(x1).reshape(b,h*w,self.head_num,c//self.head_num).permute(0,2,1,3) # b h hw c
        agent_tokens = self.agent_tokens.repeat(b,1,1) # b n c
        k,v = self.kv(agent_tokens).chunk(2,dim = -1) #
        k = k.reshape(b,self.agent_num,self.head_num,c//self.head_num).permute(0,2,3,1) #b h c n
        attn_score = ((q @ k) * self.scale).softmax(-1) # b h N n
        attn_score = self.atndp(attn_score)
        v = v.reshape(b,self.agent_num,self.head_num,c//self.head_num).permute(0,2,1,3) #b h n c
        atte = (attn_score @ v).permute(0,2,1,3).reshape(b,h,w,-1) # b h w c
        atte = self.dp(self.proj(atte))
        return atte
class ConvMambaBlock(nn.Module):
    def __init__(self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            size: int = 8,
            scan_type: str = 'scan',
            num_direction: int = 8,
            **kwargs):
        
        super().__init__()
        self.toProxy = ToProxy2(dim=hidden_dim)
        self.conv33 = DWConv(hidden_dim, hidden_dim, 3,padding=1)
        self.conv55 = DWConv(hidden_dim, hidden_dim, 5,padding=2)
        self.mlp = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, 1),nn.InstanceNorm2d(hidden_dim),nn.SiLU())  
        self.l = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.InstanceNorm2d(hidden_dim)
        self.act = nn.SiLU()
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        conv_input = x.permute(0, 3, 1, 2)
        out = self.toProxy(x)
        y33 = self.conv33(conv_input).permute(0,2,3,1) # b h w c 
        y55 = self.conv55(conv_input).permute(0,2,3,1) # b h w c
        # y35 = (y33+y55).permute(0,2,3,1)
        y33 = self.mlp(torch.cat([y33,y55],dim=-1).permute(0,3,1,2)).permute(0,2,3,1)
        
        # out = self.mlp(torch.cat([agent,y35],dim=-1).permute(0,3,1,2)).permute(0,2,3,1)#bhwc
        out = self.act(self.norm(self.l(out+y33).permute(0,3,1,2))).permute(0,2,3,1)
        out = self.l2(out)
        out = out + x
        
        return out.contiguous()
    


class LSSModule_v2(nn.Module):
    def __init__(self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            depth: int = 2,
            size: int = 8,
            scan_type: str = 'scan',
            num_direction: int = 8,
            **kwargs):
            super().__init__()
            self.convMamba = nn.ModuleList([ConvMambaBlock(hidden_dim=hidden_dim, drop_path=drop_path, norm_layer=norm_layer, attn_drop_rate=attn_drop_rate, d_state=d_state, size=size, scan_type=scan_type, num_direction=num_direction,**kwargs) for _ in range(1)])
    def forward(self, x):
        # x -  B H W C
        out = x
        for block in self.convMamba:
            out = block(out)
        return out
    

# ========== Decoder ==========
def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv2x2(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation)

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)
        return x



class LSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            size=8,
            scan_type='scan',
            num_direction=4,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        if depth % 3 == 0:
            self.blocks = nn.ModuleList([
                LSSModule_v2(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    size=size,
                    scan_type=scan_type,
                    depth=3,
                    num_direction=num_direction,
                )
                for i in range(depth)])
        elif depth % 2 == 0:
            self.blocks = nn.ModuleList([
                LSSModule_v2(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    size=size,
                    scan_type=scan_type,
                    depth=2,
                    num_direction=num_direction,
                )
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                LSSModule_v2(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    size=size,
                    scan_type=scan_type,
                    depth=2,
                    num_direction=num_direction,
                )
                for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

class MambaUPNet(nn.Module):
    def __init__(self, dims_decoder=[512, 256, 128, 64], depths_decoder=[3, 4, 6, 3],d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer = nn.LayerNorm,scan_type='scan', num_direction=4, ):
        super().__init__()
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]
        self.layers_up = nn.ModuleList()
        for i_layer in range(len(depths_decoder)):
            layer = LSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                size=8 * 2 ** (i_layer),
                scan_type=scan_type,
                num_direction=num_direction,
            )
            self.layers_up.append(layer)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in HSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, HSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = rearrange(x,'b c h w -> b h w c')
        out_features = []
        for i, layer in enumerate(self.layers_up):
            x = layer(x)
            if i != 0:
                out_features.insert(0, rearrange(x,'b h w c -> b c h w'))
        return out_features

class MFF_OCE(nn.Module):
    def __init__(self, block, layers, width_per_group = 64, norm_layer = None, ):
        super(MFF_OCE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group
        self.inplanes = 64 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 128, layers, stride=2)

        self.conv1 = conv3x3(16 * block.expansion, 32 * block.expansion, 2)
        self.bn1 = norm_layer(32 * block.expansion)
        self.conv2 = conv3x3(32 * block.expansion, 64 * block.expansion, 2)
        self.bn2 = norm_layer(64 * block.expansion)
        self.conv21 = nn.Conv2d(32 * block.expansion, 32 * block.expansion, 1)
        self.bn21 = norm_layer(32 * block.expansion)
        self.conv31 = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
        self.bn31 = norm_layer(64 * block.expansion)
        self.convf = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
        self.bnf = norm_layer(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        fpn0 = self.relu(self.bn1(self.conv1(x[0])))
        fpn1 = self.relu(self.bn21(self.conv21(x[1]))) + fpn0
        sv_features = self.relu(self.bn2(self.conv2(fpn1))) + self.relu(self.bn31(self.conv31(x[2])))
        sv_features = self.relu(self.bnf(self.convf(sv_features)))
        sv_features = self.bn_layer(sv_features)

        return sv_features.contiguous()

class MAMBAAD(nn.Module):
    def __init__(self, model_t, model_s):
        super(MAMBAAD, self).__init__()
        self.net_t = get_model(model_t)
        self.mff_oce = MFF_OCE(Bottleneck, 3)
        self.net_s = MambaUPNet(depths_decoder=model_s['depths_decoder'], scan_type=model_s['scan_type'], num_direction=model_s['num_direction'])

        self.frozen_layers = ['net_t']
        
    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs):
        feats_t = self.net_t(imgs)
        feats_t = [f.detach() for f in feats_t]
        oce_out = self.mff_oce(feats_t)  # 16 512 8 8
        b,c,h,w = oce_out.shape
        scale=20 if self.training else 0
        oce_out = add_jitter(oce_out.reshape(b,c,h*w).permute(0,2,1),scale=scale).reshape(b,h,w,c).permute(0,3,1,2).contiguous()
        feats_s = self.net_s(oce_out)
        return feats_t, feats_s

@MODEL.register_module
def mambaad(pretrained=False, **kwargs):
    model = MAMBAAD(**kwargs)
    return model

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    from util.util import get_timepc, get_net_params
    vmunet = MambaUPNet([512, 256, 128, 64], [3, 4, 6, 3])
    bs = 1
    reso = 8
    x = torch.randn(bs, 512, reso, reso).cuda()
    net = vmunet.cuda()
    net.eval()
    y = net(x)
    Flops = FlopCountAnalysis(net, x)
    print(flop_count_table(Flops, max_depth=5))
    flops = Flops.total() / bs / 1e9
    params = parameter_count(net)[''] / 1e6
    with torch.no_grad():
        pre_cnt, cnt = 5, 10
        for _ in range(pre_cnt):
            y = net(x)
        t_s = get_timepc()
        for _ in range(cnt):
            y = net(x)
        t_e = get_timepc()
    print('[GFLOPs: {:>6.3f}G]\t[Params: {:>6.3f}M]\t[Speed: {:>7.3f}]\n'.format(flops, params, bs * cnt / (t_e - t_s)))