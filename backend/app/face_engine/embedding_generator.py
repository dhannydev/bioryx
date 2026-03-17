import numpy as np
from .face_model import get_face_app


def detect_faces(frame):

    app = get_face_app()
    faces = app.get(frame)

    return faces


def generate_embedding(face):

    embedding = face.embedding

    # normalización L2 (MUY IMPORTANTE)
    embedding = embedding / np.linalg.norm(embedding)

    return embedding.astype("float32")


def get_face_pose(face):
    return face.pose


def get_face_bbox(face):
    return face.bbox