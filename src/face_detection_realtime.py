import cv2
import torch
from facenet_pytorch import MTCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MTCNN(keep_all=True, device=device)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = model.detect(frame_rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

    cv2.imshow('MTCNN PHOTO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()