from .tv import TVPrior, TVPrior3D

try:
    from .diffusion import DiffusionPrior
except ImportError:
    DiffusionPrior = None
