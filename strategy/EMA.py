import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model  # 用于 EMA 更新的模型
        self.decay = decay  # EMA 衰减系数
        self.shadow = {}
        self.device = next(model.parameters()).device  # 获取模型参数所在的设备
        self._create_ema_weights()

    def _create_ema_weights(self):
        # 初始化 EMA 权重，首先复制模型的参数并确保它们在相同设备上
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.clone().detach().to(self.device)  # 将 shadow 放到相同设备

    def update(self):
        # 更新 EMA 权重
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # 确保 param 和 shadow 在同一个设备
                self.shadow[name] = self.shadow[name].to(param.device)  # 将 shadow 更新为与 param 相同的设备
                self.shadow[name] = self.shadow[name] * self.decay + param * (1.0 - self.decay)

    def apply(self):
        # 使用 EMA 权重
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(self.shadow[name])

