# src/models/hpa_msa_bert.py

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from transformers import BertModel

# We can reuse the UEICD and CACE modules without changes
from .ueicd import UEICDModule
from .cace import CACEModule


class HPAMSA_BERT(nn.Module):
    def __init__(self, config):
        super(HPAMSA_BERT, self).__init__()
        self.config = config

        # --- 1. Text Encoder: Pre-trained BERT Model ---
        print(f"Loading pre-trained BERT model: {config.bert_model_name}")
        self.text_encoder = BertModel.from_pretrained(config.bert_model_name)

        # We still need a projection layer from BERT's hidden size to d_model
        bert_hidden_size = self.text_encoder.config.hidden_size  # Usually 768
        self.proj_t = nn.Linear(bert_hidden_size, config.d_model)

        # --- 2. Non-Text Projection Layers ---
        self.proj_v = nn.Linear(config.vision_dim, config.d_model)
        self.proj_a = nn.Linear(config.audio_dim, config.d_model)

        # --- 3. UEICD Modules ---
        self.ueicd_t = UEICDModule(config)
        self.ueicd_v = UEICDModule(config)
        self.ueicd_a = UEICDModule(config)

        # --- 4. CACE Module ---
        self.cace = CACEModule(config)

        # --- 5. Prediction Heads ---
        self.fc_out_reg = nn.Linear(config.d_model, 1)
        self.fc_out_cls = nn.Linear(config.d_model, config.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids, vision, audio):
        """
        The forward pass for the BERT-integrated HPA-MSA model.
        """
        # --- Text Feature Extraction ---
        bert_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        text_features = bert_output.last_hidden_state  # Shape: (N, L_text, 768)

        # --- Feature Projection (MFE) ---
        proj_text = self.proj_t(text_features)  # Shape: (N, L_text, d_model)
        proj_vision = self.proj_v(vision)  # Shape: (N, L_vision, d_model)
        proj_audio = self.proj_a(audio)  # Shape: (N, L_audio, d_model)

        # --- UEICD Disentanglement ---
        # Note: sequence lengths for text, vision, audio can be different.
        # UEICD and CACE are designed to handle this as they pool features.
        f_exp_text, f_imp_text = self.ueicd_t(proj_text)
        f_exp_vision, f_imp_vision = self.ueicd_v(proj_vision)
        f_exp_audio, f_imp_audio = self.ueicd_a(proj_audio)

        f_exp_list = [f_exp_text, f_exp_vision, f_exp_audio]
        f_imp_list = [f_imp_text, f_imp_vision, f_imp_audio]

        # --- CACE Fusion ---
        z_final = self.cace(f_exp_list, f_imp_list)

        # --- Prediction ---
        pred_reg = self.fc_out_reg(z_final).squeeze(-1)
        pred_cls = self.fc_out_cls(z_final)

        return {
            'pred_reg': pred_reg,
            'pred_cls': pred_cls,
            'f_exp_list': f_exp_list,
            'f_imp_list': f_imp_list
        }