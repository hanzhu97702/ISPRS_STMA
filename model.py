import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class PA(nn.Module):
    def __init__(self, dim):
        super(PA, self).__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class MlpBlock(nn.Module):
    # Feed-Forward Block
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate):
        super(MlpBlock, self).__init__()
        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out

class LinearGeneral(nn.Module):
    def __init__(self, in_dim, feat_dim):
        super(LinearGeneral, self).__init__()
        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims = dims) + self.bias
        return a

class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads, dropout_rate):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape

        q = self.query(x, dims = ([2], [0]))
        k = self.key(x, dims = ([2], [0]))
        v = self.value(x, dims = ([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim = -1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)

        out = self.out(out, dims = ([2, 3], [0, 1]))
        return out, attn_weights

def cosine_distance(x1, x2):
    """
    x1 = [b, h, n, k]
    x2 = [b, h, n, k]
    output = [b, h, n, m]
    """
    dots = torch.matmul(x1, x2)
    scale = torch.einsum('bhi, bhj -> bhij',
                         (torch.norm(x1, 2, dim=-1), torch.norm(x2, 2, dim=-2)))
    return (dots / scale)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, variant='softmax'):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.variant = variant

    def forward(self, x_q, x_kv, x):
        residual = x
        x_b, x_c, x_h, x_w = x_q.shape
        x_q = x_q.flatten(2).transpose(2,1)
        x_kv = x_kv.flatten(2).transpose(2,1)

        f_q = self.q(x_q).reshape(x_b, x_h*x_w, self.num_heads, x_c // self.num_heads).permute(0,2,1,3)
        f_kv = self.kv(x_kv).reshape(x_b, x_h*x_w, 2, self.num_heads, x_c // self.num_heads).permute(2,0,3,1,4)
        f_k, f_v = f_kv[0], f_kv[1]

        if self.variant == 'softmax':
            attn = (f_q @ f_k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
        elif self.variant == 'cosine':
            attn = cosine_distance(f_q, f_k.transpose(-2, -1))
        else:
            raise ValueError("Unknown variant value")
        attn = self.attn_drop(attn)

        out = (attn @ f_v).transpose(1,2).reshape(x_b, x_h*x_w, x_c)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(2,1).reshape(x_b, x_c, x_h, x_w)
        out = out + residual
        return out, attn

class STSA_Encoder(nn.Module):
    def __init__(self, emb_dim, emb_dim_temporal, mlp_dim, num_heads, dropout_rate, attn_dropout_rate):
        super(STSA_Encoder, self).__init__()

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim_temporal)
        self.attn_spatial = SelfAttention(emb_dim, heads = num_heads, dropout_rate = attn_dropout_rate)
        self.attn_temporal = SelfAttention(emb_dim_temporal, heads = num_heads, dropout_rate = attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm3 = nn.LayerNorm(emb_dim*2)
        self.mlp = MlpBlock(emb_dim*2, mlp_dim, emb_dim, dropout_rate)
        self.norm = nn.LayerNorm(emb_dim)

        skip_dim = int(emb_dim_temporal * 41 / emb_dim)
        self.skipcat = nn.Conv2d(skip_dim, skip_dim, [1,2], 1, 0)

    def forward(self, x1, x2):
        residual1 = x1

        out1, attn_weight1 = self.attn_spatial(self.norm1(x1))
        residual2= x2
        out2, attn_weight2 = self.attn_temporal(self.norm2(x2))
        if self.dropout:
            out1 = self.dropout(out1)
            out2 = self.dropout(out2)
        out1 += residual1
        out2 += residual2

        out2 = out2.view(out2.shape[0], -1, out1.shape[1]).transpose(1,2)

        out = torch.cat((out1, out2), dim=2)
        residual = self.skipcat(torch.concat((out1.unsqueeze(3), out2.unsqueeze(3)), dim=3)).squeeze(3)
        out = self.norm3(out)
        out = self.mlp(out)
        out += residual

        out = self.norm(out)
        return out, attn_weight1, attn_weight2

class STMA(nn.Module):
    def __init__(self, input_band, emb_dim, mlp_dim, num_heads, num_classes, attn_dropout_rate, dropout_rate):
        super(STMA, self).__init__()

        # previous convolution
        self.fe = Feat_extractor(input_band=2)
        self.conv1 = nn.Conv2d(41*2, 328, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(41*4, 328, kernel_size=1, stride=1)
        self.avg1 = nn.AvgPool2d(2, 2)
        self.avg2 = nn.AvgPool2d(1, 1)

        self.with_pos = True
        if self.with_pos:
            self.pos = PA(dim=8*input_band)
            self.pos_64 = PA(dim=2*input_band)
            self.pos_128 = PA(dim=4*input_band)

        # each transformer for each scale feature
        self.transformer1 = STSA_Encoder(
            emb_dim = 2*input_band,
            emb_dim_temporal= 32*32*2,
            mlp_dim = mlp_dim,
            num_heads = num_heads,
            dropout_rate = dropout_rate,
            attn_dropout_rate = attn_dropout_rate
        )
        self.transformer2 = STSA_Encoder(
            emb_dim = 4*input_band,
            emb_dim_temporal= 16*16*4,
            mlp_dim = mlp_dim,
            num_heads = num_heads,
            dropout_rate = dropout_rate,
            attn_dropout_rate = attn_dropout_rate
        )
        self.transformer3 = STSA_Encoder(
            emb_dim = 8*input_band,
            emb_dim_temporal= 16*16*8,
            mlp_dim = mlp_dim,
            num_heads = num_heads,
            dropout_rate = dropout_rate,
            attn_dropout_rate = attn_dropout_rate
        )
        # cross_attention
        self.cross_attention = CrossAttention(
            dim = emb_dim,
            num_heads = 1,
            qkv_bias = False,
            qk_scale = None,
            attn_drop = 0.,
            proj_drop = 0.,
            variant = 'cosine'
        )
        # decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, emb_dim//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(emb_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim//2, emb_dim//4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(emb_dim//4),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(emb_dim//4, emb_dim//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(emb_dim//4),
            nn.ReLU(inplace=True)
        )
        self.restoration = nn.Sequential(
            nn.Conv2d(emb_dim//4, emb_dim//4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(emb_dim//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim//4, num_classes, kernel_size=1)
        )
        self.seg_aux = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim//4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(emb_dim//4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(emb_dim//4, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x2, x3, x = self.fe(x, return_feat=True)
        x2_b, x2_t, x2_c, x2_h, x2_w = x2.shape
        x3_b, x3_t, x3_c, x3_h, x3_w = x3.shape
        x_b, x_t, x_c, x_h, x_w = x.shape
        if self.with_pos:
            x2 = x2.view(x2_b, x2_t*x2_c, x2_h, x2_w)
            x3 = x3.view(x3_b, x3_t*x3_c, x3_h, x3_w)
            x = x.view(x_b, x_t*x_c, x_h, x_w)

            x2 = self.pos_64(x2)
            x3 = self.pos_128(x3)
            x = self.pos(x)
        # for x2
        f1 = x2 ## [b, 41*2, 32, 32]
        x2_1 = x2.flatten(2).transpose(1,2) #[b, 1024, 84]
        x2_2 = x2.view(x2_b, x2_t, x2_c, x2_h, x2_w).flatten(2) #[b, 41, 1024*2]
        x2, attn_weight1, attn_weight2 = self.transformer1(x2_1, x2_2) # [b,1024,84]
        x2 = x2.transpose(1,2)
        x2 = x2.reshape(x2_b, x2_t*x2_c, x2_h, x2_w)
        # for x3
        f2 = x3 ## [b, 41*4, 16, 16]
        x3_1 = x3.flatten(2).transpose(1,2)  #[b, 256, 164]
        x3_2 = x3.view(x3_b, x3_t, x3_c, x3_h, x3_w).flatten(2) #[b, 41, 256*4]
        x3, _, _ = self.transformer2(x3_1, x3_2) # [b,256,164]
        x3 = x3.transpose(1,2)
        x3 = x3.reshape(x3_b, x3_t*x3_c, x3_h, x3_w)
        # for x
        high_feature = x
        f3 = x ## [b, 41*8, 16, 16]
        x_1 = x.flatten(2).transpose(1,2) #[b, 256, 328]
        x_2 = x.view(x_b, x_t, x_c, x_h, x_w).flatten(2) #[b, 41, 256*8]
        x, _, _ = self.transformer3(x_1, x_2) #[b,256,328]
        x = x.transpose(1,2)
        x = x.reshape(x_b, x_t*x_c, x_h, x_w)

        emb_x2 = self.avg1(self.conv1(x2))
        emb_x3 = self.avg2(self.conv2(x3))

        # MCA module
        feature, attn = self.cross_attention(emb_x2, emb_x3, x)
        # decoder
        aux = F.interpolate(self.seg_aux(high_feature), size=(128,128), mode='bilinear', align_corners=False)
        output = self.decoder1(f3+feature)
        output = self.decoder2(output+f1)
        output = F.interpolate(self.restoration(output), size=(128,128), mode='bilinear', align_corners=False)

        return output, aux

class Feat_extractor(nn.Module):
    def __init__(self, input_band):
        super(Feat_extractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        newconv1 = nn.Conv2d(input_band, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:2, :, :].copy_(resnet.conv1.weight.data[:, 0:2, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        self.transform1 = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.transform2 = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.transform3 = nn.Sequential(
            nn.Conv2d(256, 8, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

    def forward(self, img, return_feat=False):
        B, T, C, H, W = img.shape
        img = img.view(B*T, C, H, W)
        x0 = self.layer0(img)

        x1 = self.maxpool(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x = self.layer3(x3)

        x2 = self.transform1(x2)
        x3 = self.transform2(x3)
        x = self.transform3(x)

        x2 = x2.view(B, T, 2, x2.shape[2], x2.shape[3])
        x3 = x3.view(B, T, 4, x3.shape[2], x3.shape[3])
        x = x.view(B, T, 8, x.shape[2], x.shape[3])
        if return_feat:
            return x2, x3, x
        else:
            return x
