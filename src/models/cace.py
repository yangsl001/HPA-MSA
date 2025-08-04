# src/models/cace.py

import torch
import torch.nn as nn


class CACEModule(nn.Module):
    def __init__(self, config):
        """
        Cross-modal Adaptive Complementary Enhancement Module.
        This module performs a hierarchical, cascaded fusion of cues.

        Args:
            config (EasyDict): A configuration object containing model hyperparameters.
                               Requires d_model, cace_nhead, cace_dropout.
        """
        super(CACEModule, self).__init__()
        self.config = config

        # --- 阶段一: 显性主导融合 ---
        # 1. 可学习的 Query, 用于从显性线索中形成初步判断
        self.q_primary = nn.Parameter(torch.randn(1, 1, config.d_model))
        # 2. 注意力层
        self.attn_primary = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.cace_nhead,
            dropout=config.cace_dropout,
            batch_first=True
        )

        # --- 阶段二: 冲突感知隐性线索修正 ---
        # 1. 注意力层 (Query将是 Z_primary)
        self.attn_rectify = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.cace_nhead,
            dropout=config.cace_dropout,
            batch_first=True
        )

        # --- 最终整合 ---
        # LayerNorm 和一个简单的前馈网络 (FFN) 来稳定和增强表示
        self.layer_norm = nn.LayerNorm(config.d_model)
        # 论文中提到 "a final linear transformation to stabilize the training"
        # 我们可以用一个简单的FFN来实现这一点
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model)
        )

    def forward(self, f_exp_list, f_imp_list):
        """
        Args:
            f_exp_list (list of torch.Tensor): A list of explicit cue tensors.
                                               e.g., [f_exp_text, f_exp_vision, f_exp_audio]
            f_imp_list (list of torch.Tensor): A list of implicit cue tensors.
                                               e.g., [f_imp_text, f_imp_vision, f_imp_audio]

        Returns:
            torch.Tensor: The final fused representation Z_final (batch_size, d_model).
        """
        batch_size = f_exp_list[0].shape[0]

        # 将列表中的张量堆叠起来，形成 (batch_size, num_modalities, d_model) 的序列
        exp_cues = torch.stack(f_exp_list, dim=1)
        imp_cues = torch.stack(f_imp_list, dim=1)

        # --- 阶段一: 计算 Z_primary ---
        q_primary_batch = self.q_primary.expand(batch_size, -1, -1)
        z_primary, _ = self.attn_primary(
            query=q_primary_batch,
            key=exp_cues,
            value=exp_cues
        )  # z_primary.shape: (batch_size, 1, d_model)

        # --- 阶段二: 计算 Z_rectify ---
        # 使用 z_primary 作为 Query 去关注隐性线索
        z_rectify, _ = self.attn_rectify(
            query=z_primary,  # z_primary 已经是 (N, 1, E) 形状
            key=imp_cues,
            value=imp_cues
        )  # z_rectify.shape: (batch_size, 1, d_model)

        # --- 最终整合 ---
        # 论文公式 (13): Z_final = Linear(LayerNorm(Z_primary + Z_rectify + Σf_exp + Σf_imp))
        # 我们将所有部分相加，实现残差连接
        # .squeeze(1) 去掉多余的序列长度为1的维度
        sum_exp_cues = torch.sum(exp_cues, dim=1)
        sum_imp_cues = torch.sum(imp_cues, dim=1)

        # 整合所有信息
        integrated_info = z_primary.squeeze(1) + z_rectify.squeeze(1) + sum_exp_cues + sum_imp_cues

        # 通过 LayerNorm 和 FFN
        z_final = self.layer_norm(integrated_info)
        z_final = z_final + self.ffn(z_final)  # 再加一个残差连接

        return z_final