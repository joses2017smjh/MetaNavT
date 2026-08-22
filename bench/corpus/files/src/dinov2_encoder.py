"""DINOv2 encoder wrapper used by runs 45-49, 52, 53, 55."""

MODEL_NAME = "dinov2_vitb14"

def load_encoder(freeze_backbone: bool = True):
    """Load DINOv2 ViT-B/14. Learning rate is set in the run config, not here."""
    return MODEL_NAME
