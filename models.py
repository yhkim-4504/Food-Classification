import timm

def load_SwinV2_Base_256(weight_name: str, num_classes: int, **kwargs):
    model = timm.create_model(weight_name, pretrained=True, num_classes=num_classes)

    return model