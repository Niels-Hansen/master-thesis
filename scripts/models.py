import torch.nn as nn
from torchvision.models import efficientnet_v2_s, vit_b_16, resnext101_32x8d


class ModelFactory:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device

    def get_model(self, model_name):
            if model_name == "efficientnet_v2_s":
                model = efficientnet_v2_s(weights="DEFAULT")
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            elif model_name == "vit_b_16":
                model = vit_b_16(weights="DEFAULT")
                model.heads.head = nn.Linear(model.heads.head.in_features, self.num_classes)
            elif model_name == "resnext101_32x8d":
                model = resnext101_32x8d(weights="DEFAULT")
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            else:
                raise ValueError(f"Model {model_name} is not supported.")

            return model.to(self.device)

