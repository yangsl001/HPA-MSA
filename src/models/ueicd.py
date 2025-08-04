# src/models/ueicd.py

import torch
import torch.nn as nn


class UEICDModule(nn.Module):
    def __init__(self, config):
        """
        Unimodal Explicit-Implicit Cue Decoupling Module.
        This module uses a dual-channel Transformer architecture to disentangle cues.

        Args:
            config (EasyDict): A configuration object containing model hyperparameters.
                               Requires d_model, ueicd_nhead, ueicd_nlayers, ueicd_dropout.
        """
        super(UEICDModule, self).__init__()
        self.config = config

        # 1. 可学习的 [CLS] token，用作序列的聚合器
        # 形状为 (1, 1, d_model) 以便与 (batch_size, seq_len, d_model) 的输入拼接
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # 2. 显性通道 (Explicit Channel)
        encoder_layer_exp = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.ueicd_nhead,
            dim_feedforward=config.d_model * 4,  # 常规设置
            dropout=config.ueicd_dropout,
            activation='gelu',
            batch_first=True  # <-- 重要：确保输入形状为 (N, S, E)
        )
        self.transformer_encoder_exp = nn.TransformerEncoder(
            encoder_layer=encoder_layer_exp,
            num_layers=config.ueicd_nlayers
        )

        # 3. 隐性通道 (Implicit Channel)
        # 注意：这两个通道不共享参数
        encoder_layer_imp = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.ueicd_nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.ueicd_dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder_imp = nn.TransformerEncoder(
            encoder_layer=encoder_layer_imp,
            num_layers=config.ueicd_nlayers
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
                              This is the projected feature from a single modality.

        Returns:
            tuple: A tuple containing:
                - f_exp (torch.Tensor): Explicit cue representation (batch_size, d_model).
                - f_imp (torch.Tensor): Implicit cue representation (batch_size, d_model).
        """
        batch_size = x.shape[0]

        # 将 [CLS] token 扩展到与 batch_size 匹配
        # B, 1, D
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # 将 [CLS] token 拼接到序列的开头
        # x_with_cls.shape: (batch_size, seq_len + 1, d_model)
        x_with_cls = torch.cat((cls_tokens, x), dim=1)

        # 分别通过两个 Transformer 通道
        output_exp = self.transformer_encoder_exp(x_with_cls)
        output_imp = self.transformer_encoder_imp(x_with_cls)

        # 提取 [CLS] token 对应的输出作为最终的表征
        # [CLS] token 在序列的第0个位置
        # f_exp = output_exp[:, 0, :]  # (batch_size, d_model)
        f_exp = torch.mean(output_exp[:, 1:, :], dim=1)
        # f_imp = output_imp[:, 0, :]  # (batch_size, d_model)
        f_imp = torch.mean(output_imp[:, 1:, :], dim=1)

        return f_exp, f_imp