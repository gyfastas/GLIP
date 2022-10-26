from .generalized_rcnn import GeneralizedRCNN
from .generalized_vl_rcnn import GeneralizedVLRCNN

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,
                                 "GeneralizedVLRCNN": GeneralizedVLRCNN
                                 }


def build_detection_model(cfg): 
    """
    just build by an easy dict like mapping. 
    We can modify this to registry.
    """
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
