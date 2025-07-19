from facenet_pytorch import MTCNN
from PIL import Image
from torch import tensor
from typing import List

model4detect = MTCNN(
    image_size=224,
    thresholds=[0.5, 0.7, 0.8],
    keep_all=True
)


def get_bboxes_faces(image: Image) -> List[tensor] | None:
    boxes, _ = model4detect.detect(image)

    if boxes is not None:
        return boxes
    
    raise AssertionError('Лица не были найдены')

def get_cropped_faces(image: Image) -> List[tensor]:
    boxes = get_bboxes_faces(image)

    cropped_faces = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        w_margin = 0.75 * w
        h_margin = 0.75 * h

        new_x1 = max(0, x1 - w_margin)
        new_y1 = max(0, y1 - h_margin)
        new_x2 = min(image.width, x2 + w_margin)
        new_y2 = min(image.height, y2 + h_margin)
        cropped_faces.append(image.crop((new_x1, new_y1, new_x2, new_y2)))


    return cropped_faces