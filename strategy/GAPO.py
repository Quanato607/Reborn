import torch
import torch.nn as nn
import math
def softmax(xs, T=1.0):
    xs = list(map(float, xs))
    m = max(xs)                    # 数值稳定性：减去最大值
    exps = [math.exp((x - m)/T) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

class GAPO(nn.Module):
    def __init__(self, num_tasks=2, p_norm=2, full_proportion=0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.p_norm = p_norm

        # 可学习的 alpha 和 lambda (裁剪阈值)
        self.alpha_raw = [math.sqrt((1-full_proportion)*10)/10, math.sqrt(full_proportion*10)/10]
    def forward(self, grads1, grads2):
        # L2 归一化
        grads1 = self.l2_normalize(grads1)
        grads2 = self.l2_normalize(grads2)

        # 使用可学习的 alpha 来混合两个梯度
        alpha = softmax(self.alpha_raw)
        blended_grads = [alpha[0] * g1 + alpha[1] * g2 for g1, g2 in zip(grads1, grads2)]

        return blended_grads

    def l2_normalize(self, grads, eps=1e-12):
        flat_grads = torch.cat([g.reshape(-1) for g in grads])
        norm = torch.linalg.vector_norm(flat_grads) + eps
        return [g / norm for g in grads]

