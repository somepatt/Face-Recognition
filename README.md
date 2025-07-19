# FaceRecognition

## Описание

FaceRecognition — это модульный проект для детекции, выравнивания и распознавания лиц на изображениях и в реальном времени с использованием современных нейросетевых архитектур (MTCNN, Stacked Hourglass, ResNet50). Проект включает пайплайн для обработки изображений, обучающие ноутбуки и предобученные веса моделей.

---

## Структура проекта

- `src/`
  - `face_detection.py` — детекция лиц с помощью MTCNN.
  - `face_landmarks.py` — определение ключевых точек лица (landmarks), выравнивание лица, визуализация heatmap.
  - `face_recognition.py` — извлечение эмбеддингов и распознавание лиц на основе ResNet50.
  - `face_detection_realtime.py` — детекция лиц с веб-камеры в реальном времени.
  - `functions.py` — вспомогательные функции для визуализации.
- `pipeline/pipeline.py` — полный пайплайн: детекция → выравнивание → распознавание.
- `models/` — предобученные веса моделей:
  - `resnet50_arcfaceloss_20epoch.pth` — ResNet50 с ArcFace Loss (20 эпох).
  - `resnet50_crossentropy_20epoch.pth` — ResNet50 с CrossEntropy (20 эпох).
  - `sgh_95_epoch.pth` — Stacked Hourglass для landmarks (95 эпох).
- `notebooks/detect_and_recognition_training.ipynb` — ноутбук с кодом обучения и экспериментов.
- `requirements.txt` — список зависимостей.

---

## Быстрый старт

1. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Запуск пайплайна на изображении:**
   ```python
   from pipeline.pipeline import function
   function('path_to_image.jpg')
   ```
   Вывод: идентификаторы найденных персон.

3. **Детекция лиц в реальном времени:**
   ```bash
   python src/face_detection_realtime.py
   ```
   Для выхода нажмите `q` в окне камеры.

---

## Описание основных модулей

### Детекция лиц (`src/face_detection.py`)
- Использует MTCNN для поиска лиц на изображении.
- Функции:
  - `get_bboxes_faces(image)` — возвращает координаты лиц.
  - `get_cropped_faces(image)` — возвращает обрезанные изображения лиц с отступами.

### Ключевые точки и выравнивание (`src/face_landmarks.py`)
- Stacked Hourglass сеть для поиска 5 ключевых точек лица.
- Функции:
  - `get_key_points(image)` — возвращает координаты ключевых точек.
  - `aligned_image(image)` — возвращает выровненное по ключевым точкам лицо.
  - Визуализация heatmap и наложение точек.

### Распознавание лиц (`src/face_recognition.py`)
- ResNet50 с кастомным классификатором и ArcFace Loss.
- Функции:
  - `recognite_person(image)` — возвращает id распознанного человека.
  - `predict_embedding(image)` — возвращает эмбеддинг изображения.

### Пайплайн (`pipeline/pipeline.py`)
- Функция `function(path)`:
  1. Детектирует лица.
  2. Выравнивает каждое лицо.
  3. Распознаёт личность.
  4. Выводит id найденных персон.

---

## Обучение моделей

- Весь процесс обучения описан в ноутбуке `notebooks/detect_and_recognition_training.ipynb`:
  - Подготовка датасета (выделение лиц, выравнивание, разметка).
  - Обучение Stacked Hourglass для landmarks.
  - Обучение ResNet50 для распознавания (ArcFace Loss, CrossEntropy).
  - Визуализация результатов и тестирование пайплайна.

---

## Зависимости

Все необходимые библиотеки указаны в `requirements.txt`. Основные:
- torch, torchvision, pytorch-lightning
- facenet-pytorch
- albumentations
- opencv-python
- pillow
- matplotlib, numpy, pandas
