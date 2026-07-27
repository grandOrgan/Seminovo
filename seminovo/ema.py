"""Exponential moving averages for stable validation and inference."""

import torch


class ExponentialMovingAverage:
    """Track trainable parameters and temporarily apply their moving average."""

    def __init__(self, decay):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}

    @torch.no_grad()
    def initialize(self, module):
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }

    def align(self, module):
        parameters = dict(module.named_parameters())
        self.shadow = {
            name: value.to(device=parameters[name].device, dtype=parameters[name].dtype)
            for name, value in self.shadow.items()
            if name in parameters
        }

    @torch.no_grad()
    def update(self, module):
        if not self.shadow:
            self.initialize(module)
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def apply(self, module):
        if self.backup:
            raise RuntimeError("EMA parameters are already applied")
        self.backup = {}
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, module):
        for name, parameter in module.named_parameters():
            if name in self.backup:
                parameter.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {name: value.detach().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {name: value.detach().clone() for name, value in state.items()}
