import torch
import torch.nn as nn

class GAPO(nn.Module):
    def __init__(self, num_tasks=2, p_norm=2, init_lambda=1.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.p_norm = p_norm

        # 可学习的 alpha 和 lambda (裁剪阈值)
        self.alpha_raw = nn.Parameter(torch.zeros(num_tasks))
        self.lambda_ = nn.Parameter(torch.tensor(init_lambda))  # 初始值为 1.0

        # 初始化 alpha 和 lambda
        nn.init.normal_(self.alpha_raw, mean=0, std=0.1)

    def forward(self, grads1, grads2):
        # L2 归一化
        grads1 = self.l2_normalize(grads1)
        grads2 = self.l2_normalize(grads2)

        # 使用可学习的 alpha 来混合两个梯度
        alpha = torch.sigmoid(self.alpha_raw)
        blended_grads = [alpha[0] * g1 + alpha[1] * g2 for g1, g2 in zip(grads1, grads2)]

        # 梯度裁剪：根据 lambda 自适应裁剪梯度
        clipped_grads = self.apply_gradient_clipping(blended_grads)

        return clipped_grads

    def l2_normalize(self, grads, eps=1e-12):
        flat_grads = torch.cat([g.reshape(-1) for g in grads])
        norm = torch.linalg.vector_norm(flat_grads) + eps
        return [g / norm for g in grads]

    def apply_gradient_clipping(self, grads):
        # 计算每个梯度的 L2 范数
        for i, grad in enumerate(grads):
            grad_norm = torch.linalg.norm(grad)
            # 如果梯度的 L2 范数超过阈值 lambda_，则进行裁剪
            if grad_norm > self.lambda_:
                grads[i] = grad * (self.lambda_ / grad_norm)  # 裁剪梯度，使其范数不超过 lambda_
        return grads

