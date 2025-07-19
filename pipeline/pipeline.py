from src.face_detection import get_cropped_faces
from src.face_landmarks import aligned_image
from src.face_recognition import recognite_person
from PIL import Image


def function(path: str):
    image = Image.open(path)
    faces = get_cropped_faces(image)

    for face in faces:
        aligned_face = aligned_image(face)
        person = recognite_person(aligned_face)
        print(f'Found {person}')
    