from facenet_pytorch import MTCNN
from PIL import Image
from typing import List
from torch import tensor

model = MTCNN(
    image_size=224,
    thresholds=[0.5, 0.7, 0.8],
    keep_all=True
)

def get_bboxes_faces(image: Image) -> List[tensor] | None:
    boxes = model.detect(image)

    if boxes is not None:
        return boxes
    
    return "Модель не определила лица на фото("
