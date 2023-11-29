import torch
import torch.nn as nn
import adder
import numpy as np

# B -> Batch Size                                           400
# C -> Number of Input Channels                             1
# IH -> Image Height                                        28
# IW -> Image Width                                         28
# P -> Patch Size                                           4
# E -> Embedding Dimension                                  128
# S -> Sequence Length = IH/P * IW/P                        7*7
# Q -> Query Sequence length                                49+1
# K -> Key Sequence length                                  49+1
# V -> Value Sequence length (same as Key length)           49+1
# H -> Number of heads                                      4
# HE -> Head Embedding Dimension = E/H                      128 // 4 = 32


class EmbedLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 用卷积来划分patch
        self.conv1 = nn.Conv2d(args.n_channels, args.embed_dim, kernel_size=args.patch_size, stride=args.patch_size)    # [1, 128, 4, 4]
        # 连接上一个[B, 1, embed_dim]维的token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.embed_dim), requires_grad=True)
        # 加上一个[B, num_patches+1, embed_dim]维的pos_embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, (args.img_size // args.patch_size) ** 2 + 1, args.embed_dim), requires_grad=True) # [1, 49+1, 128]
        # Weight init
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # [B, C, IH, IW] -> [B, E, IH/P, IW/P] (Embedding the patches)
        # [400, 1, 28, 28] -> [400, 128, 7, 7]
        x = self.conv1(x)
        # flatten: [B E IH/P IW/P] -> [B, E, IH/P * IW/P] (Flattening the patches)
        #          [400, 128, 7, 7] -> [400, 128, 49]
        # transpose: [B, E, IH/P * IW/P] -> [B, IH/P * IW/P, E]
        #            [400, 128, 49] -> [400, 49, 128]
        x = x.flatten(2).transpose(1, 2)
        # 扩展出B维来，和x相加
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)   # [1, 1, 128] -> [400, 1, 128]
        x = torch.cat((cls_token, x), dim=1)        # x [400, 49, 128] -> [400, 49+1, 128]
        x = x + self.pos_embedding                  # [400, 49+1, 128] + [1, 49+1, 128], broadcast add dim 1 and 2
        return x



class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_attention_heads = args.n_attention_heads                     # 4
        self.embed_dim = args.embed_dim                                     # 128
        self.head_embed_dim = self.embed_dim // self.n_attention_heads      # 128 / 4 = 32
        self.seq_length = (args.img_size // args.patch_size) ** 2 + 1       # 50
        self.device = args.device
        
        # """Vanilla ViT"""
        # self.scale = self.head_embed_dim ** -0.5
        # self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)   # [128, 128], 所有token share
        # self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)      # [128, 128], 所有token share
        # self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)    # [128, 128], 所有token share
        
        """Adder-ViT modification"""
        self.scale = (2 * self.head_embed_dim * (1 - 2 / np.pi)) ** -0.5
        # # 用adder1x1替换全连接层
        self.queries    = conv1x1(args.embed_dim, self.head_embed_dim * self.n_attention_heads)     # [1, 128, 400, 50] -> [1, 128, 400, 50]
        self.queriesln  = nn.LayerNorm([self.embed_dim, args.batch_size, self.seq_length])          # 根据论文page3, 每个adder层后面加Layer norm
        self.keys       = conv1x1(args.embed_dim, self.head_embed_dim * self.n_attention_heads)     # [1, 128, 400, 50] -> [1, 128, 400, 50]
        self.keysln     = nn.LayerNorm([self.embed_dim, args.batch_size, self.seq_length])          # 根据论文page3, 每个adder层后面加Layer norm
        self.values     = conv1x1(args.embed_dim, self.head_embed_dim * self.n_attention_heads)     # [1, 128, 400, 50] -> [1, 128, 400, 50]
        self.valuesln   = nn.LayerNorm([self.embed_dim, args.batch_size, self.seq_length])          # 根据论文page3, 每个adder层后面加Layer norm

        self.outln      = nn.LayerNorm(self.embed_dim)
        self.outfc      = nn.Linear(self.embed_dim, self.embed_dim)                                 # W_o
        

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape     # [400, 49+1, 128]

        """Vanilla ViT"""
        # xq = self.queries(x).reshape(batch_size, seq_length, self.n_attention_heads, self.head_embed_dim)
        # xq = xq.transpose(1, 2)
        # # B, Q, E -> B, Q, H, HE
        # # [400, 50, 128] -> [400, 50, 4, 32]
        # # B, Q, H, HE -> B, H, Q, HE
        # # [400, 50, 4, 32] -> [400, 4, 50, 32]

        # xk = self.keys(x).reshape(batch_size, seq_length, self.n_attention_heads, self.head_embed_dim)  
        # xk = xk.transpose(1, 2)
        # B, K, E -> B, K, H, HE
        # [400, 50, 128] -> [400, 50, 4, 32]
        # B, K, H, HE -> B, H, K, HE
        # [400, 50, 4, 32] -> [400, 4, 50, 32]

        # xv = self.values(x).reshape(batch_size, seq_length, self.n_attention_heads, self.head_embed_dim)
        # xv = xv.transpose(1, 2)
        # B, V, E -> B, V, H, HE
        # [400, 50, 128] -> [400, 50, 4, 32]
        # B, V, H, HE -> B, H, V, HE
        # [400, 50, 4, 32] -> [400, 4, 50, 32]


        """Adder-ViT modification"""
        xq = x.reshape(1, embed_dim, batch_size, seq_length)    # [400, 49+1, 128] -> [1, 128, 400, 50]
        xq = self.queries(xq)                                   # [1, 128, 400, 50] -> [1, 128, 400, 50]
        xq = self.queriesln(xq)
        xq = xq.reshape(batch_size, seq_length, self.n_attention_heads, self.head_embed_dim)    # [1, 128, 400, 50] -> [400, 50, 4, 32]
        xq = xq.transpose(1, 2)                                 # [400, 50, 4, 32] -> [400, 4, 50, 32]

        xk = x.reshape(1, embed_dim, batch_size, seq_length)    # [400, 49+1, 128] -> [1, 128, 400, 50]
        xk = self.keys(xk)                                      # [1, 128, 400, 50] -> [1, 128, 400, 50]
        xk = self.keysln(xk)
        xk = xk.reshape(batch_size, seq_length, self.n_attention_heads, self.head_embed_dim)    # [1, 128, 400, 50] -> [400, 50, 4, 32]
        xk = xk.transpose(1, 2)                                 # [400, 50, 4, 32] -> [400, 4, 50, 32]

        xv = x.reshape(1, embed_dim, batch_size, seq_length)    # [400, 49+1, 128] -> [1, 128, 400, 50]
        xv = self.values(xv)                                    # [1, 128, 400, 50] -> [1, 128, 400, 50]
        xv = self.valuesln(xv)
        xv = xv.reshape(batch_size, seq_length, self.n_attention_heads, self.head_embed_dim)    # [1, 128, 400, 50] -> [400, 50, 4, 32]
        xv = xv.transpose(1, 2)                                 # [400, 50, 4, 32] -> [400, 4, 50, 32]


        xq = xq.reshape([-1, seq_length, self.head_embed_dim])      # B, H, Q, HE -> (BH), Q, HE     [400, 4, 50, 32] -> [1600, 50, 32]
        xk = xk.reshape([-1, seq_length, self.head_embed_dim])      # B, H, K, HE -> (BH), K, HE     [400, 4, 50, 32] -> [1600, 50, 32]
        xv = xv.reshape([-1, seq_length, self.head_embed_dim])      # B, H, V, HE -> (BH), V, HE     [400, 4, 50, 32] -> [1600, 50, 32]

        xk = xk.transpose(1, 2)                                     # (BH), K, HE -> (BH), HE, K    [1600, 50, 32] -> [1600, 32, 50]
        x_attention = xq.bmm(xk)                                    # (BH), Q, HE  .  (BH), HE, K -> (BH), Q, K     [1600, 50, 32]*[1600, 32, 50] -> [1600, 50, 50]
        x_attention = x_attention * self.scale
        x_attention = torch.softmax(x_attention, dim=-1)
        x_attention = x_attention + torch.eye(x_attention.shape[-1]).to(self.device)    # 论文公式(20)

        x = x_attention.bmm(xv)                                                         # (BH), Q, K . (BH), V, HE -> (BH), Q, HE  [1600, 50, 50]*[1600, 50, 32] -> [1600, 50, 32]
        x = x.reshape([-1, self.n_attention_heads, seq_length, self.head_embed_dim])    # (BH), Q, HE -> B, H, Q, HE   [1600, 50, 32] -> [400, 4, 50, 32]
        x = x.transpose(1, 2)                                                           # B, H, Q, HE -> B, Q, H, HE     [400, 4, 50, 32] -> [400, 50, 4, 32]
        x = x.reshape(batch_size, seq_length, embed_dim)                                # B, Q, H, HE -> B, Q, E        [400, 50, 4, 32] -> [400, 50, 128]
        x = self.outln(x)
        x = self.outfc(x)
        return x



class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = nn.LayerNorm(args.embed_dim)
        self.attn = Attention(args)
        self.norm2 = nn.LayerNorm(args.embed_dim)
        self.activation = nn.GELU()

        # self.fc1 = nn.Linear(args.embed_dim, args.embed_dim * args.forward_mul)   # [400, 50, 128] -> [400, 50, 256]
        # self.fc2 = nn.Linear(args.embed_dim * args.forward_mul, args.embed_dim)   # [400, 50, 256] -> [400, 50, 128]

        """Adder-ViT modification"""
        # 用adder1x1替换MLP
        self.fc1 = conv1x1(args.embed_dim, args.embed_dim * args.forward_mul)       # [1, 128, 400, 50] -> [1, 256, 400, 50]
        self.fc1ln = nn.LayerNorm(args.embed_dim * args.forward_mul)                # 根据论文page3, 每个adder层后面加Layer norm
        # 用adder1x1替换MLP
        self.fc2 = conv1x1(args.embed_dim * args.forward_mul, args.embed_dim)       # [1, 256, 400, 50] -> [1, 128, 400, 50]
        self.fc2ln = nn.LayerNorm(args.embed_dim)                                   # 根据论文page3, 每个adder层后面加Layer norm

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape                 # [400, 49+1, 128]

        x = x + self.attn(self.norm1(x))                            # Skip connections
        x1 = self.norm2(x)

        x1 = x1.reshape(1, embed_dim, batch_size, seq_length)       # [400, 49+1, 128] -> [1, 128, 400, 50]
        x1 = self.fc1(x1)                                           # [1, 128, 400, 50] -> [1, 256, 400, 50]
        x1 = x1.reshape(batch_size, seq_length, -1)                 # [1, 256, 400, 50] -> [400, 50, 256]
        x1 = self.fc1ln(x1)

        x1 = self.activation(x1)
        x1 = x1.reshape(1, -1, batch_size, seq_length)              # [400, 50, 256] -> [1, 256, 400, 50]
        x1 = self.fc2(x1)                                           # [1, 256, 400, 50] -> [1, 128, 400, 50]
        x1 = x1.reshape(batch_size, seq_length, -1)                 # [1, 128, 400, 50] -> [400, 50, 128]
        x1 = self.fc2ln(x1)
        
        x = x + x1
        return x



class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(args.embed_dim, args.n_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Get CLS token
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x



class VisionTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = EmbedLayer(args)
        self.encoder = nn.Sequential(*[Encoder(args) for _ in range(args.n_layers)], nn.LayerNorm(args.embed_dim))
        self.norm = nn.LayerNorm(args.embed_dim) # Final normalization layer after the last block
        self.classifier = Classifier(args)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x




def conv1x1(in_planes, out_planes):
    " 1x1 convolution for FFN layers "
    return adder.adder2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

