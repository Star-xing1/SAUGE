"""
The architecture of basic SAUGE (vitb version).
"""
import torch
import torch.nn as nn
from segment_anything import SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import numbers
from einops import rearrange
import torch.nn.functional as F
class Sam_Generator(nn.Module):
    def __init__(
        self,
        sam_generator: SamAutomaticMaskGenerator,
    ) -> None:
        super(Sam_Generator, self).__init__()
        self.model = sam_generator

    def forward(self, image):
        return self.model.generate(image)
    
    def get_img_features(self):
        return self.model.predictor.model.image_encoder.blocks_outputs
    
    def get_mask_embedding(self):
        return self.model.predictor.model.mask_decoder.mask_decoder_output_embedding


class Self_Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
    def forward(self, x):
        return self.attention(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias, groups=dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)


    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)


        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.proj_dropout(out)
        return out
    

## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.drop_out = nn.Dropout(0.1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = self.drop_out(x)
        return x


class FFB(nn.Module):
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(FFB, self).__init__()

        self.conv2 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):
        input_S = self.conv2(input_S)
        input_R = self.norm1(input_R)
        input_S = self.norm1(input_S)
        input_R = input_R + self.attn(input_R, input_S)
        output = input_R + self.ffn(self.norm2(input_R))

        return output


class Gradient_Net(torch.nn.Module):
  def __init__(self, in_channels=192):
    super(Gradient_Net, self).__init__()

    self.in_channels = in_channels
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

    sobel_x = sobel_x.repeat(self.in_channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(self.in_channels, 1, 1, 1)

    self.weight_x = torch.nn.Parameter(data=sobel_x, requires_grad=False)
    self.weight_y = torch.nn.Parameter(data=sobel_y, requires_grad=False)

  def forward(self, x):
    grad_x = F.conv2d(x, self.weight_x, padding=1, groups=self.in_channels)
    grad_y = F.conv2d(x, self.weight_y, padding=1, groups=self.in_channels)
    gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    return gradient



class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d,upsampling)


class STN_Block(nn.Module):
    def __init__(self, in_channels,LayerNorm_type="WithBias"):
        super(STN_Block, self).__init__()
        # BCHW->B(C/4)(2H)(2W)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=2, stride=2),
            LayerNorm(in_channels // 4, LayerNorm_type),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels // 4,
                in_channels // 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm(in_channels // 4, LayerNorm_type),
            nn.GELU()
        )
    def forward(self, x):
        x_upsample = self.output_upscaling(x)
        out = x_upsample + self.proj(x_upsample)
        return out


class STN(nn.Module):
    def __init__(self, in_channels,LayerNorm_type="WithBias"):
        super(STN, self).__init__()
        self.dblk_img_embd = STN_Block(256)
        self.dblk_img_shallow_1 = STN_Block(in_channels)
        self.dblk_img_shallow_2 = STN_Block(in_channels // 4)
        self.img_embd_shallow_fuse = FFB(in_channels // 16, in_channels // 16)
        self.mask_fuse = FFB(48, in_channels // 16)
        self.proj_side_1 = nn.Sequential(
            nn.Conv2d(in_channels // 16, in_channels // 32, kernel_size=3, padding=1),
            LayerNorm(in_channels // 32, LayerNorm_type),
            nn.GELU(),
            nn.Conv2d(in_channels // 32, in_channels // 64, kernel_size=1),
            LayerNorm(in_channels // 64, LayerNorm_type),
            nn.GELU(),
        )
        self.proj_embd = nn.Sequential(
            nn.Conv2d(192, in_channels // 8, kernel_size=3, padding=1),
            LayerNorm(in_channels // 8, LayerNorm_type),
            nn.GELU(),
            nn.Conv2d(in_channels // 8, 48, kernel_size=3, padding=1),
            LayerNorm(48, LayerNorm_type),
            nn.GELU(),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, in_channels // 16, kernel_size=2, stride=2),
            LayerNorm(in_channels // 16, LayerNorm_type),
            nn.GELU()
        )
        self.proj_side_2 = nn.Sequential(
            nn.Conv2d(in_channels // 16, in_channels // 32, kernel_size=3, padding=1),
            LayerNorm(in_channels // 32, LayerNorm_type),
            nn.GELU(),
            nn.Conv2d(in_channels // 32, in_channels // 64, kernel_size=1),
            LayerNorm(in_channels // 64, LayerNorm_type),
            nn.GELU(),
        )
        self.proj_side_3 = nn.Sequential(
            nn.Conv2d(in_channels // 16, in_channels // 32, kernel_size=3, padding=1),
            LayerNorm(in_channels // 32, LayerNorm_type),
            nn.GELU(),
            nn.Conv2d(in_channels // 32, in_channels // 64, kernel_size=1),
            LayerNorm(in_channels // 64, LayerNorm_type),
            nn.GELU(),
        )
        self.proj_gate_1 = nn.Sequential(
            nn.Conv2d(in_channels // 64, 1, kernel_size=3, padding=1),
        )
        self.proj_gate_2 = nn.Sequential(
            nn.Conv2d(in_channels // 64, 1, kernel_size=3, padding=1),
        )



    def forward(self, *x, img_size, mask_embd, sam_predictor):

        img_feat_shallow = x[0]
        img_embd = x[1]
        mask_embd = self.proj_embd(mask_embd)

        img_embd = self.dblk_img_embd(img_embd)
        img_embd = self.upsample(img_embd)
        side_output_1 = self.proj_side_1(img_embd)
        gate_1 = torch.sigmoid(self.proj_gate_1(side_output_1))
        side_output_1 = sam_predictor.model.postprocess_masks(side_output_1, sam_predictor.input_size, img_size)
        img_embd = gate_1 * img_embd

        img_feat_shallow = self.dblk_img_shallow_1(img_feat_shallow)
        img_feat_shallow = self.dblk_img_shallow_2(img_feat_shallow)
        img_feat_fuse = self.img_embd_shallow_fuse(img_embd, img_feat_shallow)

        side_output_2 = self.proj_side_2(img_feat_fuse)
        gate_2 = torch.sigmoid(self.proj_gate_2(side_output_2))
        side_output_2 = sam_predictor.model.postprocess_masks(side_output_2, sam_predictor.input_size, img_size)
        img_feat_fuse = gate_2 * img_feat_fuse

        # fuse
        img_mask_fuse = self.mask_fuse(img_feat_fuse, mask_embd)
        img_mask_fuse = sam_predictor.model.postprocess_masks(img_mask_fuse, sam_predictor.input_size, img_size)
        side_output_3 = self.proj_side_3(img_mask_fuse)

        return side_output_1, side_output_2, side_output_3
        


class SAUGE(nn.Module):
    def __init__(self, args, sam_generator=None, classes=1, mode='train'):
        super(SAUGE, self).__init__()
        
        self.sam_generator= Sam_Generator(sam_generator=sam_generator)

        self.STN = STN(in_channels=768)

        self.segmentation_head = SegmentationHead(
            in_channels=12,
            out_channels=classes,
            kernel_size=1
        )

        self.proj_mask = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            LayerNorm(16, LayerNorm_type="WithBias"),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            LayerNorm(3, LayerNorm_type="WithBias"),
            nn.GELU(),
        )

        self.res_fuse_1 = nn.Sequential(
            nn.Conv2d(24, 12, kernel_size=1)
        )
        self.res_fuse_2 = nn.Sequential(
            nn.Conv2d(24, 12, kernel_size=1)
        )
        self.res_fuse_3 = nn.Sequential(
            nn.Conv2d(36, 12, kernel_size=1)
        )

        self.grad = Gradient_Net(in_channels=768 // 4)

        self.args=args
        self.mode = mode

    def forward(self, x):
        img_H, img_W = x.shape[2], x.shape[3]

        x = x.cpu()
        x = x.permute(0, 2, 3, 1)
        x_np = x.numpy()
        mask_list = []
        mask_embd_list = []
        img_feat_list = []
        img_feat = []
        for i in range(x_np.shape[0]):
            single_img = x_np[i]
            single_img = (single_img * 255).astype(np.uint8)
            img_mask = self.sam_generator(single_img)
            if self.mode == 'train':
                masks = [torch.zeros(single_img.shape[0], single_img.shape[1], 1, device='cuda')]
            
                for mask_data in img_mask:
                    mask = (mask_data["segmentation"] * 1)[:, :, np.newaxis]
                    mask = mask.astype(np.uint8)
                    mask = torch.from_numpy(mask).to('cuda')
                    masks.append(mask)

                mask_np = torch.cat(masks, dim=2).permute(2, 0, 1).float()
                padded_mask = torch.zeros(192, mask_np.size(1), mask_np.size(2), device='cuda')

                if mask_np.size(0) > 192:
                    mask = mask_np[:192]
                else:
                    padded_mask[:mask_np.size(0)] = mask_np
                    mask = padded_mask
                
                mask = mask.unsqueeze(0)
                mask_list.append(mask)
            
            img_feat_list.append(self.sam_generator.get_img_features())
            m_embd = self.sam_generator.get_mask_embedding()
            m_embd = self.proj_mask(m_embd)
            m_embd = m_embd.unsqueeze(0)
            m_embd = rearrange(m_embd, 'b head c h w -> b (head c) h w')
            mask_embd_list.append(m_embd)

        for i in range(len(img_feat_list[0])):
            tensors_to_merge = [l[i] for l in img_feat_list]
            merged_tensor = torch.cat(tensors_to_merge, dim=0)
            img_feat.append(merged_tensor)
        if self.mode == 'train':
            mask_feat = torch.cat(mask_list, dim=0)
        mask_embd = torch.cat(mask_embd_list, dim=0)
        features_in = img_feat

        # output
        side_output_1, side_output_2, side_output_3 = self.STN(*features_in, img_size=(img_H, img_W), mask_embd=mask_embd, sam_predictor=self.sam_generator.model.predictor)

        mask_edge, mask_edge_count = None, None
        if self.mode == 'train':
            mask_grad = self.grad(mask_feat)
            mask_grad = mask_grad > 0
            mask_edge = torch.sum(mask_grad, dim=1, keepdim=True)
            mask_edge_count = mask_edge
            mask_edge = mask_edge > 0
            mask_edge = mask_edge.float()
            mask_edge_count = mask_edge_count.float()
        side_res_1 = torch.sigmoid(self.segmentation_head(side_output_1))
        side_output_2 = self.res_fuse_1(torch.cat([side_output_1, side_output_2], dim=1))
        side_res_2 = torch.sigmoid(self.segmentation_head(side_output_2))
        side_output_3 = self.res_fuse_2(torch.cat([side_output_2, side_output_3], dim=1))
        side_res_3 = torch.sigmoid(self.segmentation_head(side_output_3))

        multi_outputs = self.res_fuse_3(torch.cat([side_output_1, side_output_2, side_output_3], dim=1))
        outputs = [side_res_1, side_res_2, side_res_3]
        final_output = torch.sigmoid(self.segmentation_head(multi_outputs))

        return outputs, final_output, mask_edge, mask_edge_count


