import torch

class EMA:
    def __init__(self, model, decay=0.99):
        self.model = model  # 用于 EMA 更新的模型
        self.decay = decay  # EMA 衰减系数
        self.shadow = {}
        self._create_ema_weights()

    def _create_ema_weights(self):
        # 初始化 EMA 权重，首先复制模型的参数
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.clone().detach()

    def update(self):
        # 更新 EMA 权重
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.shadow[name] = self.shadow[name] * self.decay + param * (1.0 - self.decay)

    def apply(self):
        # 使用 EMA 权重
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(self.shadow[name])
