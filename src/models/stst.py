import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from src.models.basemodel import TorchModel 

class CAW_MASA_STST(TorchModel):
    def __init__(self, n_classes, n_channels, n_time_points, n_spe_channels, n_bands, device='cpu'):
        super(CAW_MASA_STST, self).__init__(n_classes=n_classes, n_channels=n_channels, n_time_points=n_time_points, n_spe_channels=n_spe_channels, n_bands=n_bands, device=device)

    def set_model_parameters(self, n_classes, n_channels, n_time_points, n_spe_channels, n_bands):
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_time_points = n_time_points
        self.n_bands = n_bands
        self.n_spe_channels = n_spe_channels
    
        self.masa = self.MASA(self.n_channels, self.n_time_points, self.n_spe_channels, self.n_bands)

        self.spectral_conv_block_2 = nn.Sequential(
            ConvBlock(40,30,(1,13)),
            ConvBlock(30,10,(1,11)),
        )
        self.spectral_avg_pool=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,8)),
            nn.Dropout2d(0.5)
        )
        self.spatial_conv_block_1 = ConvBlock(1,40,(self.n_channels, 1))

        self.spatial_conv_block_2 = nn.Sequential(
            ConvBlock(40,30,(1,13)),
            ConvBlock(30,10,(1,11)),
        )
        self.spatial_avg_pool=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,8)),
            nn.Dropout2d(0.5)
        )

        self.fusion_conv_block=nn.Sequential(
            ConvBlock(80,70,(1,13)),
            ConvBlock(70,80,(1,11)),
            nn.AdaptiveAvgPool2d((1,8)),
            nn.Dropout2d(0.5)
        )

        self.stst = self.STSTransformerBlock(40,40)
       
        self.caw = self.CAW(self.n_spe_channels, 2)

        self.feature_length = (100) * 8

        self.fc = nn.Linear(self.feature_length, self.n_classes)
        self.flatten = nn.Flatten()

    def forward(self, x, x_cwt):
        x_cwt, _ = self.caw(x_cwt)

        #MASA
        x_spe = self.masa(x_cwt)

        x_spe_out = self.spectral_conv_block_2(x_spe)
        x_spe_out = self.spectral_avg_pool(x_spe_out).squeeze()

        x_spa = self.spatial_conv_block_1(x.unsqueeze(1))
        x_spa_out = self.spatial_conv_block_2(x_spa)
        x_spa_out = self.spatial_avg_pool(x_spa_out)
        x_spa_out = x_spa_out.squeeze()

        fuse_out = self.stst(x_spe, x_spa)
        fuse_out = self.fusion_conv_block(fuse_out).squeeze()

        out = torch.cat((x_spe_out, x_spa_out, fuse_out),dim=1)

        out = self.fc(self.flatten(out))
        
        return out
    
    class MASA(nn.Module):
        def __init__(self, n_channels, n_time_points, n_spe_channels, n_bands):
            super().__init__()
            self.n_channels = n_channels
            self.n_time_points = n_time_points
            self.n_spe_channels = n_spe_channels
            self.n_bands = n_bands
            self.spe_width = n_spe_channels // n_bands

            self.asa = nn.ModuleList([nn.Sequential(
                ConvBlock(n_channels, n_spe_channels, (self.spe_width, 1)),
                Rearrange("a b c d -> a (c d) b"),
                self.GCN(n_spe_channels, n_time_points),
                Rearrange("a b (c d) -> a c d b", d=1),
                ConvBlock(n_spe_channels, 2, (1,1))) for _ in range(self.n_bands)])
            
            self.spectral_conv_block_1 = ConvBlock(n_channels, 30, (n_spe_channels, 1))
        
        def forward(self, x_cwt):
            x_cwts = x_cwt.chunk(self.n_bands, dim=2)
            
            x_asf = torch.cat([self.asa[i](x_cwts[i]) for i in range(self.n_bands)], dim=1)

            x_spe= self.spectral_conv_block_1(x_cwt)
            x_spe = torch.cat((x_spe, x_asf), dim=1)

            return x_spe
        
        class GCN(nn.Module):
            def __init__(self, n_channels_spe, n_time_points):
                super().__init__() 

                self.a = nn.Parameter(torch.rand((n_time_points, n_time_points)))
                self.k = 2
                self.n_filters = n_channels_spe
                self.Theta = nn.Parameter(torch.randn((self.k, self.n_filters, self.n_filters)))
                self.adj = None
                self.device = self.a.device

            def forward(self, x):
                
                x = x.squeeze(axis=2)
                b, c, l = x.size()
                feature_matrix = x

                # Similarity Matrix
                self.diff = (x.expand([c,b,c,l]).permute(2,1,0,3)-x).permute(1,0,2,3)
                self.diff=torch.abs(self.diff).sum(-1)
                self.diff=F.normalize(self.diff,dim=0)
                tmpS = torch.exp(torch.relu((1-self.diff)*self.a))

                # Laplacian matrix 
                self.adj = tmpS / torch.sum(tmpS,axis=1,keepdims=True)
                D = torch.diag_embed(torch.sum(self.adj,axis=1))
                L = D - self.adj

                # Chebyshev graph convolution
                firstOrder=torch.eye(c, device=x.device)
                lambda_max = 2.0
                L_t = (2 * L) / lambda_max - firstOrder
                cheb_polynomials = [firstOrder, L_t]
                for i in range(2, self.k):
                    cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
                    
                output = torch.zeros(b, c, self.n_filters, device=x.device)
                for kk in range(self.k):
                    T_k = cheb_polynomials[kk].expand([b,c,c])
                    rhs = torch.bmm(T_k.permute(0, 2, 1), feature_matrix)
                    output = output + torch.matmul(rhs, self.Theta[kk])
                output=torch.relu(output)
                return output
        
    class STSTransformerBlock(nn.Module):
        def __init__(self, emb_size1,emb_size2,num_heads=5,drop_p=0.5,forward_expansion=4,forward_drop_p=0.5):
            super().__init__()
            self.emb_size = emb_size1
            self.att_drop1 = nn.Dropout(drop_p)
            self.projection1 = nn.Linear(emb_size1, emb_size1)
            self.projection2 = nn.Linear(emb_size1, emb_size1)
            self.drop1=nn.Dropout(drop_p)
            self.drop2=nn.Dropout(drop_p)

            self.layerNorm1=nn.LayerNorm(emb_size1)
            self.layerNorm2=nn.LayerNorm(emb_size2)
    
            self.queries1 = nn.Linear(emb_size1, emb_size1)
            self.values1 = nn.Linear(emb_size1, emb_size1)
            self.keys2 = nn.Linear(emb_size2, emb_size2)
            self.values2 = nn.Linear(emb_size2, emb_size2)

            self.layerNorm3 = nn.LayerNorm(emb_size1+emb_size2)
            self.mha = self.MultiHeadAttention(emb_size1+emb_size2, num_heads, 0.5)
            self.drop3=nn.Dropout(drop_p)

            self.ffb=nn.Sequential(
                nn.LayerNorm(emb_size1+emb_size2),
                self.FeedForwardBlock(
                    emb_size1+emb_size2, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )

        def forward(self, x1, x2):
            x1=rearrange(x1, 'b e (h) (w) -> b (h w) e ')
            x2=rearrange(x2, 'b e (h) (w) -> b (h w) e ')
            res1=x1
            res2=x2

            x1 = self.layerNorm1(x1)
            x2 = self.layerNorm2(x2)
            queries1 = self.queries1(x1) 
            values1 = self.values1(x1)
            keys2 = self.keys2(x2)
            values2 = self.values2(x2)

            energy = torch.einsum('bqd, bkd -> bqk', keys2, queries1)
            scaling = self.emb_size ** (1 / 2)
            att = F.softmax(energy / scaling, dim=-1)
            att = self.att_drop1(att)

            out1 = torch.einsum('bal, blv -> bav ', att, values1)
            out1 = self.projection1(out1)
            x1 = self.drop1(out1)
            x1+=res1

            out2 = torch.einsum('bal, blv -> bav ', att, values2)
            out2 = self.projection2(out2)
            x2 = self.drop2(out2)
            x2+=res2

            x=torch.cat((x1,x2),dim=-1)
            res = x
            x=self.layerNorm3(x)
            x=self.mha(x)
            x=self.drop3(x)
            x += res

            res = x
            x = self.ffb(x)
            x += res
            x = rearrange(x, 'b t e -> b e 1 t')
            return x
        
        class MultiHeadAttention(nn.Module):
            def __init__(self, emb_size, num_heads, dropout):
                super().__init__()
                self.emb_size = emb_size
                self.num_heads = num_heads
                self.keys = nn.Linear(emb_size, emb_size)
                self.queries = nn.Linear(emb_size, emb_size)
                self.values = nn.Linear(emb_size, emb_size)
                self.att_drop = nn.Dropout(dropout)
                self.projection = nn.Linear(emb_size, emb_size)
                self.scaling = self.emb_size ** (1 / 2)

            def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
                queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
                keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
                values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
                energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) 
                if mask is not None:
                    fill_value = torch.finfo(torch.float32).min
                    energy.mask_fill(~mask, fill_value)

                att = F.softmax(energy / self.scaling, dim=-1)
                att = self.att_drop(att)
                out = torch.einsum('bhal, bhlv -> bhav ', att, values)
                out = rearrange(out, "b h n d -> b n (h d)") 
                out = self.projection(out)
                return out
            
        class FeedForwardBlock(nn.Sequential):
            def __init__(self, emb_size, expansion, drop_p):
                super().__init__(
                    nn.Linear(emb_size, expansion * emb_size),
                    nn.GELU(),
                    nn.Dropout(drop_p),
                    nn.Linear(expansion * emb_size, emb_size),
                )

    class CAW(nn.Module):
        def __init__(self, channel, reduction = 1):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ELU(inplace  = True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            if len(x.shape)==3:
                b, c, t = x.size()
                xstd=((x-x.mean(-1).view(b,c,1))**2)
                xstd = F.normalize(xstd.sum(-1),dim=-1)
                attn = self.fc(xstd).view(b, c, 1)
            else:
                b, s, c, t = x.size()
                x = rearrange(x, 'b s c t -> b c s t')
                xstd=((x-x.mean(-1).view(b,c,s,1))**2)
                xstd = F.normalize(xstd.sum(-1),dim=-1)
                attn = self.fc(xstd).view(b, c, s, 1)
            out = x * attn.expand_as(x)
            return out, attn


        
        
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, with_bn=True, with_relu=True, stride=1, padding=0, bias=True):
        super().__init__()
        self.with_bn=with_bn
        self.with_relu=with_relu
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,stride=stride,padding=padding,bias=bias)
        self.batchNorm=None
        self.relu=None
        
        if with_bn:
            self.batchNorm=nn.BatchNorm2d(out_channels)
        if with_relu:
            self.relu=nn.ELU()

    def forward(self, x):
        out=self.conv2d(x)
        if self.with_bn:
            out=self.batchNorm(out)
        if self.with_relu:
            out=self.relu(out)
        return out