import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer
import math
from torch.nn.utils import spectral_norm

class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(Encoder, self).__init__()
        # Downsample
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(base_channels)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(base_channels * 2) 
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(base_channels * 4)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(base_channels * 8)
        self.conv5 = nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=4, stride=2, padding=1)
        self.norm5 = nn.InstanceNorm2d(base_channels * 16)
        self.conv6 = nn.Conv2d(base_channels * 16, base_channels * 32, kernel_size=4, stride=2, padding=1)
        self.norm6 = nn.InstanceNorm2d(base_channels * 32)

        # 1*1 for adjust channel
        self.adjust1 = nn.Conv2d(base_channels * 32, 512, kernel_size=1)
        self.adjust2 = nn.Conv2d(base_channels * 16, 512, kernel_size=1)
        self.adjust3 = nn.Conv2d(base_channels * 8, 512, kernel_size=1)

        # 3*3 conv after addition
        self.conv_after_add1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_after_add2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        c0 = F.relu(self.norm1(self.conv1(x)))  # 256 * 256 * 64
        c1 = F.relu(self.norm2(self.conv2(c0))) # 128 * 128 * 128
        c2 = F.relu(self.norm3(self.conv3(c1))) # 64 * 64 * 256
        c3 = F.relu(self.norm4(self.conv4(c2))) # 32 * 32 * 512
        c4 = F.relu(self.norm5(self.conv5(c3))) # 16 * 16 * 1024
        c5 = F.relu(self.norm6(self.conv6(c4))) # 8 * 8 * 2048

        p5 = self.adjust1(c5)              
        p4 = self.adjust2(c4)  
        p3 = self.adjust3(c3)  

        up_p5 = self.upsample(p5)                
        merged_p4 = F.relu(self.conv_after_add1(up_p5 + p4))  # Element-wise addition followed by 3x3 conv              
        up_merged_p4 = self.upsample(merged_p4)   
        merged_p3 = F.relu(self.conv_after_add2(up_merged_p4 + p3))  # Element-wise addition followed by 3x3 conv

        return merged_p3

# class Encoder(nn.Module):
#     def __init__(self, in_channels, base_channels):
#         super(Encoder, self).__init__()
#         # Downsample
#         self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
#         self.norm1 = nn.InstanceNorm2d(base_channels)
#         self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
#         self.norm2 = nn.InstanceNorm2d(base_channels * 2) 
#         self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
#         self.norm3 = nn.InstanceNorm2d(base_channels * 4)
#         self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1)
#         self.norm4 = nn.InstanceNorm2d(base_channels * 8)

#     def forward(self, x): # x: (256, 256, 3)
#         c0 = F.relu(self.norm1(self.conv1(x)))  # x -> (256, 256, 64)
#         c1 = F.relu(self.norm2(self.conv2(c0))) # x -> (128, 128, 128)
#         c2 = F.relu(self.norm3(self.conv3(c1))) # x -> (64, 64, 256) 
#         c3 = F.relu(self.norm4(self.conv4(c2))) # x -> (32, 32, 512)

#         return c3

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Decoder1(nn.Module):
    def __init__(self, in_channels, ):
        super(Decoder1, self).__init__()
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1) 
        self.norm1 = nn.InstanceNorm2d(in_channels // 2)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(in_channels // 4)
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(in_channels // 4, in_channels // 8, 3, padding=1)
        self.norm3 = nn.InstanceNorm2d(in_channels // 8)
        
        self.conv_final = nn.Conv2d(in_channels // 8, 3, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(self.upsample1(x))))
        x = self.relu(self.norm2(self.conv2(self.upsample2(x))))
        x = self.relu(self.norm3(self.conv3(self.upsample3(x))))
        x = self.conv_final(x)
        return x

class Decoder2(nn.Module):
    def __init__(self, in_channels):
        super(Decoder2, self).__init__()
        
        self.upsample1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(in_channels // 2)
        self.se1 = SELayer(in_channels // 2)
        
        self.upsample2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(in_channels // 4)
        self.se2 = SELayer(in_channels // 4)
        
        self.upsample3 = nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(in_channels // 8)
        self.se3 = SELayer(in_channels // 8)
        
        self.conv_final = nn.Conv2d(in_channels // 8, 3, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.norm1(self.upsample1(x)))
        x = self.se1(x)
        x = self.relu(self.norm2(self.upsample2(x)))
        x = self.se2(x)
        x = self.relu(self.norm3(self.upsample3(x)))
        x = self.se3(x)
        x = self.conv_final(x)
        return x


class Decoder3(nn.Module):
    def __init__(self, in_channels):
        super(Decoder3, self).__init__()
        
        self.upsample1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 2)
        
        self.upsample2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        
        self.upsample3 = nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.BatchNorm2d(in_channels // 8)
        
        self.conv_final = nn.Conv2d(in_channels // 8, 3, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.norm1(self.upsample1(x)))
        x = self.relu(self.norm2(self.upsample2(x)))
        x = self.relu(self.norm3(self.upsample3(x)))
        x = self.conv_final(x)
        return x



class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.custom_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.custom_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, tgt, memory_key, memory_value, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 下面所有的shape都是(1024, b, 512)
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Custom attention, where keys and values come from different sources
        tmp = self.custom_attn1(tgt, memory_key, memory_value, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt2 = tmp[0] # return: attn_output, attn_output_weights
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)


        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm4(tgt)
        
        return tgt

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width, num_frames=3, frame_emb_scale=0.001):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        self.frame_emb_scale = frame_emb_scale

        x_pos = torch.arange(width).unsqueeze(0).repeat(height, 1).float() # (32, 32)
        y_pos = torch.arange(height).unsqueeze(1).repeat(1, width).float() # (32, 32)
        
        pe = torch.zeros(d_model, height, width) # (512, 32, 32)
        

        for i in range(0, d_model, 4): 
            if i + 3 >= d_model:    continue  

            div_term = 10000 ** (2 * (i // 2) / d_model)

            pe[i, :, :] = torch.sin(x_pos / div_term)
            pe[i+1, :, :] = torch.cos(x_pos / div_term)

            pe[i+2, :, :] = torch.sin(y_pos / div_term)
            pe[i+3, :, :] = torch.cos(y_pos / div_term)

        pe = 0.0001 * pe
        # print(pe[0].max(), pe[0].min())

        self.register_buffer('pe', pe)
        # Frame number embedding
        self.frame_embedding = nn.Embedding(num_frames, d_model)

    def forward(self, x, frame_number=None):
        device = self.frame_embedding.weight.device
        if frame_number is None: # for target
            frame_number = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            frame_number = frame_number.to(device)
        frame_emb = self.frame_embedding(frame_number).unsqueeze(2).unsqueeze(3)  # (batch_size, d_model, 1, 1)
        frame_emb = frame_emb * self.frame_emb_scale
        x = x + self.pe.unsqueeze(0) + frame_emb
        return x



class ImageColorizationTransformer(nn.Module):
    def __init__(self, sketch_dim, color_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len=1024):
        super(ImageColorizationTransformer, self).__init__()
        self.sketch_encoder = Encoder(in_channels=1, base_channels=64)
        self.color_encoder = Encoder(in_channels=3, base_channels=64)
        # self.output_decoder = Decoder1(in_channels=512)
        self.output_decoder = Decoder2(in_channels=512)
        # self.output_decoder = Decoder3(in_channels=512)
        self.pos_encoder = PositionalEncoding2D(512, 32, 32)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers)
        
        # Custom Transformer Decoder Layer
        self.custom_decoder_layer = CustomDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        
        # Transformer Decoder
        self.transformer_decoder = nn.ModuleList([self.custom_decoder_layer for _ in range(num_decoder_layers)])

    def forward(self, target_sketch, ref1_sketch, ref1_color, ref2_sketch, ref2_color):
        # target_sketch: (batch, 1, 256, 256)
        target_sketch_emb = self.sketch_encoder(target_sketch) # (batch, 512, 32, 32)
        ref1_sketch_emb = self.sketch_encoder(ref1_sketch)
        ref1_color_emb = self.color_encoder(ref1_color)
        ref2_sketch_emb = self.sketch_encoder(ref2_sketch)
        ref2_color_emb =self.color_encoder(ref2_color)

        batch_size = target_sketch_emb.size(0)
        frame_numbers_ref1 = torch.tensor([1] * batch_size).cuda()
        frame_numbers_ref2 = torch.tensor([2] * batch_size).cuda()

        target_sketch_emb = self.pos_encoder(target_sketch_emb)
        ref1_sketch_emb = self.pos_encoder(ref1_sketch_emb, frame_numbers_ref1)
        ref1_color_emb = self.pos_encoder(ref1_color_emb, frame_numbers_ref1)
        ref2_sketch_emb = self.pos_encoder(ref2_sketch_emb, frame_numbers_ref2)
        ref2_color_emb =self.pos_encoder(ref2_color_emb, frame_numbers_ref2)

        target_sketch_flat = target_sketch_emb.flatten(start_dim=2)
        ref1_sketch_flat = ref1_sketch_emb.flatten(start_dim=2)
        ref1_color_flat = ref1_color_emb.flatten(start_dim=2)
        ref2_sketch_flat = ref2_sketch_emb.flatten(start_dim=2)
        ref2_color_flat = ref2_color_emb.flatten(start_dim=2)

        target_sketch_input = target_sketch_flat.permute(2, 0, 1)
        ref1_sketch_input = ref1_sketch_flat.permute(2, 0, 1)
        ref1_color_input = ref1_color_flat.permute(2, 0, 1)
        ref2_sketch_input = ref2_sketch_flat.permute(2, 0, 1)
        ref2_color_input = ref2_color_flat.permute(2, 0, 1)

        sketch_k1 = self.transformer_encoder(ref1_sketch_input)
        color_v1 = self.transformer_encoder(ref1_color_input)
        sketch_k2 = self.transformer_encoder(ref2_sketch_input)
        color_v2 = self.transformer_encoder(ref2_color_input)

        sketch_k_concat = torch.cat([sketch_k1, sketch_k2], dim=0)
        color_v_concat = torch.cat([color_v1, color_v2], dim=0)

        output = target_sketch_input
        for layer in self.transformer_decoder:
            output = layer(output, sketch_k_concat, color_v_concat)

        output = output.permute(1, 2, 0)

        batch_size = output.size(0)
        output = output.view(batch_size, 512, 32, 32)

        final_output = self.output_decoder(output) 

        return final_output
        

