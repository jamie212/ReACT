import torch
import torch.nn.functional as F
from torchvision.models import vgg19
import numpy as np
import cv2

def get_vgg_features(model, x, layers):
    features = []
    for name, layer in model.features._modules.items():
        x = layer(x)
        if str(name) in layers:
            features.append(x)
    return features

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def perceptual_style_loss(vgg_model, target, output, layer_indices, lambda_p, lambda_s):
    layers = [str(idx) for idx in layer_indices]
    target_features = get_vgg_features(vgg_model, target, layers)
    output_features = get_vgg_features(vgg_model, output, layers)
    p_loss = sum(F.l1_loss(t_feat, o_feat) for t_feat, o_feat in zip(target_features, output_features))
    s_loss = sum(F.l1_loss(gram_matrix(t_feat), gram_matrix(o_feat)) for t_feat, o_feat in zip(target_features, output_features))
    return p_loss, s_loss

