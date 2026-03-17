import cv2

from backend.app.face_engine.embedding_generator import (
    detect_faces,
    generate_embedding
)

from backend.app.face_engine.face_database import load_embeddings
from backend.app.face_engine.faiss_index import FaceIndex


def run_face_recognition():

    embeddings, labels = load_embeddings()

    index = FaceIndex()

    index.build(embeddings, labels)

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        faces = detect_faces(frame)

        for face in faces:

            embedding = generate_embedding(face)

            results = index.search(embedding)

            best_match = results[0]

            name = best_match["label"]
            score = best_match["score"]

            print(name, score)

            x1, y1, x2, y2 = face.bbox.astype(int)

            # threshold correcto para ArcFace
            if score > 0.5:
                label = f"{name} ({score:.2f})"
                color = (0, 255, 0)
            else:
                label = "Desconocido"
                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        cv2.imshow("Reconocimiento Facial", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()