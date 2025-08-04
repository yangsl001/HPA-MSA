# src/models/hpa_msa.py

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from .ueicd import UEICDModule
from .cace import CACEModule


# 在这个阶段，我们先不导入 UEICD 和 CACE，因为它们还未创建
# from .ueicd import UEICDModule
# from .cace import CACEModule

class HPAMSA(nn.Module):
    def __init__(self, config):
        """
        Args:
            config (EasyDict): A configuration object containing model hyperparameters.
        """
        super(HPAMSA, self).__init__()
        self.config = config

        # --- 步骤 1: 特征投影层 (MFE Part) ---
        # Dynamically create projection layers based on config

        self.proj_type = config.get('proj_type', 'linear')

        if self.proj_type == 'conv':
            print("Using Conv1D for feature projection.")
            # For Conv1D, input is (N, C_in, L), output is (N, C_out, L_out)
            self.proj_t = nn.Conv1d(config.text_dim, config.d_model, kernel_size=config.proj_kernel_size_t)
            self.proj_v = nn.Conv1d(config.vision_dim, config.d_model, kernel_size=config.proj_kernel_size_v)
            self.proj_a = nn.Conv1d(config.audio_dim, config.d_model, kernel_size=config.proj_kernel_size_a)
        else:  # Default to linear
            print("Using nn.Linear for feature projection.")
            self.proj_t = nn.Linear(config.text_dim, config.d_model)
            self.proj_v = nn.Linear(config.vision_dim, config.d_model)
            self.proj_a = nn.Linear(config.audio_dim, config.d_model)

        # --- 步骤 2: 实例化 UEICD 模块 ---
        # 为每个模态创建一个独立的 UEICD 实例
        self.ueicd_t = UEICDModule(config)
        self.ueicd_v = UEICDModule(config)
        self.ueicd_a = UEICDModule(config)
        # --- 步骤 3: 实例化 CACE 模块 ---
        self.cace = CACEModule(config)

        # --- 步骤 4: 添加预测头 ---
        # 回归头 (输出一个连续值)
        self.fc_out_reg = nn.Linear(config.d_model, 1)

        # 分类头 (输出每个类别的 logits)
        # 假设 config 中有 num_classes 参数
        self.fc_out_cls = nn.Linear(config.d_model, config.num_classes)

    def forward(self, text, vision, audio):
        """
        Args:
            text (torch.Tensor): Text features (batch_size, seq_len, text_dim)
            vision (torch.Tensor): Vision features (batch_size, seq_len, vision_dim)
            audio (torch.Tensor): Audio features (batch_size, seq_len, audio_dim)

        Returns:
            A dictionary containing the projected features.
            In later steps, this will return the final model output.
        """
        # --- 步骤 1: 应用投影层 ---
        if self.proj_type == 'conv':
            # Input shape: (N, L, C). Conv1D needs (N, C, L).
            text = text.transpose(1, 2)
            vision = vision.transpose(1, 2)
            audio = audio.transpose(1, 2)

            # Apply convolution
            proj_text = self.proj_t(text)
            proj_vision = self.proj_v(vision)
            proj_audio = self.proj_a(audio)

            # Convert back to (N, L_out, C_out) for the rest of the model
            proj_text = proj_text.transpose(1, 2)
            proj_vision = proj_vision.transpose(1, 2)
            proj_audio = proj_audio.transpose(1, 2)
        else:
            # Original linear projection
            proj_text = self.proj_t(text)
            proj_vision = self.proj_v(vision)
            proj_audio = self.proj_a(audio)

        # --- 步骤 2: 将投影后的特征送入各自的 UEICD 模块 ---
        f_exp_text, f_imp_text = self.ueicd_t(proj_text)
        f_exp_vision, f_imp_vision = self.ueicd_v(proj_vision)
        f_exp_audio, f_imp_audio = self.ueicd_a(proj_audio)
        # --- 步骤 3: 将解耦后的特征送入 CACE 模块进行融合 ---
        f_exp_list = [f_exp_text, f_exp_vision, f_exp_audio]
        f_imp_list = [f_imp_text, f_imp_vision, f_imp_audio]

        z_final = self.cace(f_exp_list, f_imp_list)

        # --- 步骤 4: 通过预测头得到最终输出 ---
        # 回归任务预测
        pred_reg = self.fc_out_reg(z_final).squeeze(-1)  # squeeze掉最后的维度1

        # 分类任务预测
        pred_cls = self.fc_out_cls(z_final)

        # 返回一个包含所有必要信息的字典
        # 训练时需要用到中间特征来计算解耦损失
        return {
            'pred_reg': pred_reg,
            'pred_cls': pred_cls,
            'f_exp_list': f_exp_list,
            'f_imp_list': f_imp_list
        }