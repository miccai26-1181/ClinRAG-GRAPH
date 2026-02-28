from typing import Dict
import torch
import torch.nn.functional as F


def compute_losses(
    out: Dict[str, torch.Tensor],
    y_true: torch.Tensor,             
    center_id: torch.Tensor,          
    lambda_adv: float,
) -> Dict[str, torch.Tensor]:

    y_logit = out["y_logit"]

    # pCR (binary) — weighted BCE (keep original definition)
    pos_weight = y_logit.new_tensor([3])  # pos_weight = N_neg / N_pos
    L_pcr = F.binary_cross_entropy_with_logits(
        y_logit,
        y_true.float(),
        pos_weight=pos_weight
    )

    adv_logit = out.get("adv_logit", None)
    if adv_logit is None:
        raise KeyError("Model output must contain 'adv_logit' for adversarial domain loss.")
    L_adv = F.cross_entropy(adv_logit, center_id.long())

    L_total = L_pcr + float(lambda_adv) * L_adv

    return {
        "L_total": L_total,
        "L_pcr": L_pcr.detach(),
        "L_adv": L_adv.detach(),
    }
