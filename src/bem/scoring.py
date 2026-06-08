from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def robust_cosine_similarity(fm1_pooled: torch.Tensor | None, fm2_pooled: torch.Tensor | None) -> float:
    """Cosine similarity between pooled detector features."""
    if fm1_pooled is None or fm2_pooled is None:
        return 0.0
    a = fm1_pooled.view(fm1_pooled.size(0), fm1_pooled.size(1))
    b = fm2_pooled.view(fm2_pooled.size(0), fm2_pooled.size(1))
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    return float((a * b).sum(dim=1).mean())


def inv_sigmoid(p, eps=1e-6):
    p = float(np.clip(p, eps, 1.0-eps))
    return np.log(p/(1.0-p))


def _sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def apply_logit(scores, cos_val, alpha, gamma = 0.01):
    # 로그릿 공간에서 감산 → 고신뢰 보존, 저신뢰 강한 억제
    # 순위 가중(w_low)로 "애매한" 박스를 더 깎음
    #cos_val = (1 - cos_val)
    
    cos_val = 1/max(cos_val, 0.01)
    ranks = np.argsort(np.argsort(scores))
    pct   = (ranks + 1) / (len(scores) + 1.0)
    w_low = (1.0 - pct)
    coef = alpha/gamma
    z     = np.array([inv_sigmoid(s) for s in scores])
    z    -= coef * cos_val * w_low# 잠깐 늘리는걸로 바꿈
    return np.array([_sigmoid(v) for v in z])


def rescore(scores: np.ndarray, cos_val: float, alpha: float = 0.6, gamma: float = 1.0) -> np.ndarray:
    if cos_val <= 0.0 or alpha <= 0.0:
        return scores
    return apply_logit(scores, cos_val, alpha, gamma)
