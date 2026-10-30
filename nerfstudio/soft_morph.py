import torch
import torch.nn.functional as F

class SoftDilation:
    def __init__(self):
        pass

    def __call__(self,
        mask: torch.Tensor,
        core: torch.Tensor,
        iters: int,) -> torch.Tensor:
        g = core
        for _ in range(iters):
            g = torch.min(
                mask,
                F.max_pool2d(g, 3, stride=1, padding=1)
            )
        return g


class SoftErode:
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return -F.max_pool2d(-x, k, stride=1, padding=k // 2)



