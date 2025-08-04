# src/utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class HPAMSALoss(nn.Module):
    """
    Computes the total loss for the HPA-MSA model.
    The total loss is a combination of task-specific losses and a disentanglement loss.
    """

    def __init__(self, config):
        """
        Args:
            config (EasyDict): Configuration object containing loss weights.
                               Requires lambda_ortho and lambda_mi.
        """
        super(HPAMSALoss, self).__init__()
        self.config = config

        # Task-specific loss functions
        self.regression_loss = nn.L1Loss()

        # Initialize CrossEntropyLoss with the label_smoothing parameter from the config
        # If 'label_smoothing' is not in config or is 0.0, it behaves like standard CrossEntropyLoss.
        smoothing = config.get('label_smoothing', 0.0)
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=smoothing)

        if smoothing > 0.0:
            print(f"Using CrossEntropyLoss with label smoothing: {smoothing}")
        # We add a config flag to control this behavior
        self.train_with_regression_only = config.get('train_with_regression_only', False)
        if self.train_with_regression_only:
            print("INFO: Training with REGRESSION loss ONLY.")
    def _compute_disentanglement_loss(self, f_exp_list, f_imp_list):
        """
        Computes the disentanglement loss (L_dis), which consists of:
        - L_ortho: Orthogonality constraint.
        - L_mi: Mutual Information minimization (approximated by distance).
        """
        # --- L_ortho: Orthogonality Constraint ---
        # Formula (8): Pushes the cosine similarity between explicit and implicit cues to 0.
        l_ortho = 0.0
        for f_exp, f_imp in zip(f_exp_list, f_imp_list):
            f_exp_norm = F.normalize(f_exp, p=2, dim=1)
            f_imp_norm = F.normalize(f_imp, p=2, dim=1)
            cos_sim = torch.sum(f_exp_norm * f_imp_norm, dim=1)
            l_ortho += (cos_sim ** 2).mean()

        l_ortho /= len(f_exp_list)
        l_mi = 0.0
        for f_exp, f_imp in zip(f_exp_list, f_imp_list):
            dist_sq = torch.sum((f_exp - f_imp) ** 2, dim=1)
            l_mi += (-dist_sq).mean()

        l_mi /= len(f_exp_list)
        l_mi_revised = 0.0
        epsilon = 1e-8  # To prevent division by zero
        for f_exp, f_imp in zip(f_exp_list, f_imp_list):
            dist_sq = torch.sum((f_exp - f_imp) ** 2, dim=1)
            # This loss is between 0 and 1. We want to minimize it.
            l_mi_revised += (1.0 / (dist_sq + 1.0)).mean()

        l_mi_revised /= len(f_exp_list)

        # Combine the two losses with weights from the config
        l_dis = self.config.lambda_ortho * l_ortho + self.config.lambda_mi * l_mi_revised

        # Return the revised mi loss for logging
        return l_dis, l_ortho, l_mi_revised

    def forward(self, model_output, labels):
        """
        Args:
            model_output (dict): The dictionary returned by the HPAMSA model.
                                 Must contain 'pred_reg', 'pred_cls', 'f_exp_list', 'f_imp_list'.
            labels (dict): A dictionary containing the ground truth labels.
                           Must contain 'regression' and 'classification'.

        Returns:
            dict: A dictionary containing all computed losses:
                  'total', 'task', 'disentangle', 'regression', 'classification', 'ortho', 'mi'.
        """
        # --- 1. Compute Task Loss (L_task) ---
        pred_reg = model_output['pred_reg']
        pred_cls = model_output['pred_cls']

        true_reg = labels['regression']
        true_cls = labels['classification']

        loss_reg = self.regression_loss(pred_reg, true_reg)
        loss_cls = self.classification_loss(pred_cls, true_cls)

        # Decide which losses contribute to the task loss for backpropagation
        if self.train_with_regression_only:
            l_task = loss_reg
        else:
            # Original behavior
            l_task = loss_reg + loss_cls

        # --- 2. Compute Disentanglement Loss (L_dis) ---
        f_exp_list = model_output['f_exp_list']
        f_imp_list = model_output['f_imp_list']

        l_dis, l_ortho, l_mi = self._compute_disentanglement_loss(f_exp_list, f_imp_list)

        # --- 3. Compute Total Loss ---
        l_total = l_task + l_dis

        return {
            'total': l_total,
            'task': l_task,
            'disentangle': l_dis,
            'regression': loss_reg,
            'classification': loss_cls,
            'ortho': l_ortho,
            'mi': l_mi
        }