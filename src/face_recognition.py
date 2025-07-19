from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch
import torch.functional as F
from torch import Tensor
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

num_classes = 1000

class ResNetEmbed(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_layer = nn.Linear(2048, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.embed_layer(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class ResNetCustom(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super().__init__()
        self.embed_net = ResNetEmbed(embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        x = self.embed_net(x)
        return x
    
    def predict(self, x):
        embed = self.forward(x)
        embed = F.normalize(embed, dim=1)
        weights = F.normalize(self.classifier.weight, dim=1)
        logits = F.linear(embed, weights)
        return torch.argmax(logits, dim=1)
    
    def predict_embed(self, x):
        return self.forward(x)


model4recognition = ResNetCustom(num_classes).to(device)


def recognite_person(image: Image) -> int:
    image_tensor = transform(image)
    predict = model4recognition.predict(image_tensor)
    return predict.item()

def predict_embedding(image: Image) -> Tensor:
    image_tensor = transform(image)
    predict = model4recognition.predict_embed(image_tensor)
    return predict