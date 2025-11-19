import torch
from math import pi, log
from functools import wraps
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Reduce

# Attention implementation     
class Attention_3(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.3):# layers,0.1
        super(Attention_3, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.attention_mlp = nn.Sequential()
        self.attention_mlp.add_module('attention_mlp', nn.Linear(in_features=dim*3, out_features=hidden_dim))
        self.attention_mlp.add_module('attention_mlp_dropout', nn.Dropout(dropout))
        self.attention_mlp.add_module('attention_mlp_activation', nn.ReLU())
        self.fc_att = nn.Linear(hidden_dim, 3)
    
    def forward(self, feas1, feas2, feas3):
        multi_hidden1 = torch.cat([feas1, feas2, feas3], dim=1) # [bsz, 768*2]
        attention = self.attention_mlp(multi_hidden1) # [bsz, 64]  
        attention = self.fc_att(attention)# [bsz, 2]
        attention = torch.unsqueeze(attention, 2) * self.scale # [bsz, 2, 1]
        attention = attention.softmax(dim = 1)
        multi_hidden2 = torch.stack([feas1, feas2, feas3], dim=2) # [bsz, 768, 2]
        fused_feat = torch.matmul(multi_hidden2, attention) # 
        fused_feat = fused_feat.squeeze() # [bsz, 64]
        fused_feat = fused_feat.view(-1,self.dim)
        return fused_feat

class Attention_2(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.3):# layers,0.1
        super(Attention_2, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        # self.weights_threshold = weights_threshold
        self.attention_mlp = nn.Sequential()
        self.attention_mlp.add_module('attention_mlp', nn.Linear(in_features=dim*2, out_features=hidden_dim))
        self.attention_mlp.add_module('attention_mlp_dropout', nn.Dropout(dropout))
        self.attention_mlp.add_module('attention_mlp_activation', nn.ReLU())
        self.fc_att = nn.Linear(hidden_dim, 2)

    def forward(self, feas1, feas2):
        multi_hidden1 = torch.cat([feas1, feas2], dim=1)
        attention = self.attention_mlp(multi_hidden1) 
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) * self.scale
        attention = attention.softmax(dim = 1)
        multi_hidden2 = torch.stack([feas1, feas2], dim=2)
        fused_feat = torch.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze()
        fused_feat = fused_feat.view(-1,self.dim)
        return fused_feat

# Attention implementation     
class Attention_Uncertainty_2(nn.Module):
    def __init__(self, dim, hidden_dim, LB, UB, dropout=0.3):# layers,0.1
        super(Attention_Uncertainty_2, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.LB = LB
        self.UB = UB
        self.attention_mlp = nn.Sequential()
        self.attention_mlp.add_module('attention_mlp', nn.Linear(in_features=dim*2, out_features=hidden_dim))
        self.attention_mlp.add_module('attention_mlp_dropout', nn.Dropout(dropout))
        self.attention_mlp.add_module('attention_mlp_activation', nn.ReLU())
        self.fc_att = nn.Linear(hidden_dim, 2)
        
    def compute_weights_from_variances(self, variances, LB, UB):
        # average variance of each sample
        mean_variances = variances.mean(dim=1)
        # compute the weights based on variances, higher variances 
        weights = 1.0 / (mean_variances + 1e-8)  # avoid divided by zero
        # set the lower and upper bound
        weights = torch.clamp(weights, min=LB, max=UB)
        weights_expanded = torch.cat([torch.ones_like(weights).unsqueeze(1), weights.unsqueeze(1)], dim=1)
        return mean_variances, weights_expanded

    def forward(self, feas1, feas2, variances, LB, UB):
        multi_hidden1 = torch.cat([feas1, feas2], dim=1)
        attention = self.attention_mlp(multi_hidden1) 
        attention = self.fc_att(attention)
        mean_variances, weights_expanded = self.compute_weights_from_variances(variances, self.LB, self.UB)
        attention = attention * weights_expanded
        attention = torch.unsqueeze(attention, 2) * self.scale
        attention = attention.softmax(dim = 1)
        multi_hidden2 = torch.stack([feas1, feas2], dim=2)
        fused_feat = torch.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze()
        fused_feat = fused_feat.view(-1,self.dim)
        return fused_feat


# Attention implementation     
class Attention_Uncertainty_3(nn.Module):
    # Fusion based on the variations of the VIB outputs
    def __init__(self, dim, hidden_dim, dropout=0.3):# layers,0.1
        super(Attention_Uncertainty_3, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.attention_mlp = nn.Sequential()
        self.attention_mlp.add_module('attention_mlp', nn.Linear(in_features=dim*3, out_features=hidden_dim))
        self.attention_mlp.add_module('attention_mlp_dropout', nn.Dropout(dropout))
        self.attention_mlp.add_module('attention_mlp_activation', nn.ReLU())
        self.fc_att = nn.Linear(hidden_dim, 3)

    def compute_weights_from_variances(self,variances1, variances2, variances3):
        # average variance of each sample
        mean_variances1 = variances1.mean(dim=1)
        mean_variances2 = variances2.mean(dim=1)
        mean_variances3 = variances3.mean(dim=1)

        weights_1 = 1.0 / (mean_variances1 + 1e-8)
        weights_2 = 1.0 / (mean_variances2 + 1e-8)
        weights_3 = 1.0 / (mean_variances3 + 1e-8)
        combined_weights = torch.stack([weights_1, weights_2, weights_3], dim=1)
        normalized_weights = F.softmax(combined_weights, dim=1)
        return normalized_weights
    
    def forward(self, feas1, feas2, feas3, variances1, variances2, variances3):
        multi_hidden1 = torch.cat([feas1, feas2, feas3], dim=1) # [bsz, 768*2]
        attention = self.attention_mlp(multi_hidden1) # [bsz, 64]  
        attention = self.fc_att(attention)# [bsz, 2]
        weights = self.compute_weights_from_variances(variances1, variances2, variances3)
        attention = attention * weights
        attention = torch.unsqueeze(attention, 2) * self.scale # [bsz, 2, 1]
        attention = attention.softmax(dim = 1)
        multi_hidden2 = torch.stack([feas1, feas2, feas3], dim=2) # [bsz, 768, 2]
        fused_feat = torch.matmul(multi_hidden2, attention) # 
        fused_feat = fused_feat.squeeze() # [bsz, 64]
        fused_feat = fused_feat.view(-1,self.dim)
        return fused_feat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)#
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 5, dim_head = 64, dropout = 0.):#
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

## perceiver cross-transformer
def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

class LatentTransformer(nn.Module):# may have promblems
    def __init__(self, num_latents, latent_dim, input_dim, depth, heads, dim_head, latent_heads, latent_dim_head,  dropout=0.,attn_dropout=0.,ff_dropout=0., weight_tie_layers=False,self_per_cross_attn=1):
    # This implementation is adapted from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
# helpers
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.cross_heads = heads
        self.cross_dim_head = dim_head
        self.depth = depth
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))# here bsz?
        self.attn_dropout = dropout
        self.ff_dropout = ff_dropout
        self.latent_heads = latent_heads
        self.latent_dim_head = latent_dim_head
        get_cross_attn = lambda: PreNorm(self.latent_dim, Attention(self.latent_dim, self.input_dim, heads = self.cross_heads, dim_head = self.cross_dim_head, dropout = self.attn_dropout), context_dim = self.input_dim)
        get_cross_ff = lambda: PreNorm(self.latent_dim, FeedForward(self.latent_dim, dropout = self.ff_dropout))
        get_latent_attn = lambda: PreNorm(self.latent_dim, Attention(self.latent_dim, heads = self.latent_heads, dim_head = self.latent_dim_head, dropout = self.attn_dropout))
        get_latent_ff = lambda: PreNorm(self.latent_dim, FeedForward(self.latent_dim, dropout = self.ff_dropout))
        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))
        
        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_embedds = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim)
        )
    def forward(self, data, mask=None, return_embeddings=False):
        b = data.shape[0]
        len = data.shape[1]
        data = rearrange(data, 'b ... d -> b (...) d')
        x = repeat(self.latents, 'n d -> b n d', b = b)# b
        # layers
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x
            
            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x
            
        if return_embeddings:
            return x

        return self.to_embedds(x)


class Encoder_Layer(nn.Module):
    def __init__(self, input_dim, depth, heads, dim_head, latent_heads, latent_dim_head, dropout=0.,attn_dropout=0.,ff_dropout=0., weight_tie_layers=False,self_per_cross_attn=1):
    # This implementation is adapted from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
        super().__init__()
        self.latent_dim = input_dim
        self.input_dim = input_dim
        self.cross_heads = heads
        self.cross_dim_head = dim_head
        self.depth = depth
        self.attn_dropout = dropout
        self.ff_dropout = ff_dropout
        self.latent_heads = latent_heads
        self.latent_dim_head = latent_dim_head
        get_cross_attn = lambda: PreNorm(self.latent_dim, Attention(self.latent_dim, self.input_dim, heads = self.cross_heads, dim_head = self.cross_dim_head, dropout = self.attn_dropout), context_dim = self.input_dim)
        get_cross_ff = lambda: PreNorm(self.latent_dim, FeedForward(self.latent_dim, dropout = self.ff_dropout))
        get_cross_attn, get_cross_ff = map(cache_fn, (get_cross_attn, get_cross_ff))
        
        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args)
            ]))

        self.to_embedds = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(self.latent_dim)
        )
        
    def forward(self, domin, non_domin, mask=None, return_embeddings=False):
        # the non-domin may have different lengths
        b = non_domin.shape[0]
        len = non_domin.shape[1]
        d = non_domin.shape[2]
        non_domin = rearrange(non_domin, 'b ... d -> b (...) d')
        x = domin

        for cross_attn, cross_ff in self.layers:
            x = cross_attn(x, context = non_domin, mask = mask) + x
            x = cross_ff(x) + x
            
        if return_embeddings:
            return x

        return self.to_embedds(x)


class VIB_estimate(nn.Module):
    # modified from https://github.com/1Konny/VIB-pytorch/blob/master/model.py
    # please reference paper: DEEP VARIATIONAL INFORMATION BOTTLENECK in ICLR 2017
    def __init__(self, dim=128, encoding_dim=64, output_dim=1):
        # dim: input dimension, X
        # encoding_dim, dimensions of the encoding Z
        
        super(VIB_estimate, self).__init__()
        self.K = encoding_dim
        self.dim = dim
        self.output_dim = output_dim

        self.encode = nn.Sequential(
            nn.Linear(self.dim, dim*2),
            nn.ReLU(True),
            nn.Linear(dim*2, dim*2),
            nn.ReLU(True),
            nn.Linear(dim*2, 2*self.K))

        self.decode = nn.Sequential(
                nn.Linear(self.K, self.output_dim))# why 10

    def forward(self, x, num_sample=1):
        if x.dim() > 2 : x = x.view(x.size(0),-1)

        statistics = self.encode(x)
        mu = statistics[:,:self.K]
        std = F.softplus(statistics[:,self.K:]-5,beta=1)

        encoding = self.reparametrize_n(mu,std,num_sample)
        logits = self.decode(encoding)

        if num_sample == 1 : pass
        elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

        kl_loss = self.compute_kl_loss(mu, std)

        return (mu, std), encoding, logits, kl_loss

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)
            
        eps = std.data.new(std.size()).normal_()
        eps = eps.to(std.device)

        return mu + eps * std

    def compute_kl_loss(self, mu, std):
        # MEAN ACROSS BATCH
        KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
        # COMPUTATION WITHIN SAMPLE
        # KL = 0.5 * (mu.pow(2) + std.pow(2) - 2*std.log() - 1)
        return KL
    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()