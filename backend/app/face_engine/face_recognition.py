import cv2

from backend.app.face_engine.embedding_generator import (
    detect_faces,
    generate_embedding
)

from backend.app.face_engine.face_database import load_embeddings
from backend.app.face_engine.faiss_index import FaceIndex


THRESHOLD = 0.55


def run_face_recognition():

    index = FaceIndex()

    # Intentar cargar índice existente
    if not index.load():
        print("Construyendo índice por primera vez...")
        embeddings, labels = load_embeddings()
        index.build(embeddings, labels)
    else:
        print("Índice cargado desde disco")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("No se pudo abrir la cámara")

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        faces = detect_faces(frame)

        for face in faces:

            embedding = generate_embedding(face)

            results = index.search(embedding)

            if len(results) == 0:
                continue

            best_match = results[0]

            name = best_match["label"]
            score = best_match["score"]

            print(name, score)

            x1, y1, x2, y2 = face.bbox.astype(int)

            if score > THRESHOLD:
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