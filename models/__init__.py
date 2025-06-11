from .npiseg3d import build_npiseg3d

from .criterion import build_mask_criterion

def build_model(args):
    return build_npiseg3d(args)

def build_criterion(args):
    return build_mask_criterion(args)