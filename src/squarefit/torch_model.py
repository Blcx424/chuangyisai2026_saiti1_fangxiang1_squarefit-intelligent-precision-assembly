"""单一六维逆向补偿网络。

网络只做一件事：从允许部署获得的特征直接输出六维补偿命令。间隙、碰撞和
残余计算都由 ``torch_geometry.py`` 的不可训练物理层负责，故这里不建立第二个
“正向间隙神经网络”。
"""

from __future__ import annotations

from torch import Tensor, nn


class InverseCompensationMLP(nn.Module):
    """轻量 MLP：标准化输入 -> 标准化六维补偿命令。"""

    def __init__(self, input_size: int, *, hidden_size: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size 与 hidden_size 都必须为正整数。")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 内。")
        layers: list[nn.Module] = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, 6))
        self.network = nn.Sequential(*layers)

    def forward(self, normalized_features: Tensor) -> Tensor:
        if normalized_features.ndim != 2:
            raise ValueError("网络输入必须为二维 [batch, feature] 张量。")
        return self.network(normalized_features)
