import cv2
import time

from backend.app.face_engine.embedding_generator import (
    detect_faces,
    generate_embedding
)

from backend.app.face_engine.face_database import load_embeddings
from backend.app.face_engine.faiss_index import FaceIndex


THRESHOLD = 0.55

#CONFIGURACIÓN
FRAME_SKIP = 3
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480


def run_face_recognition():

    index = FaceIndex()

    #Cargar índice
    if not index.load():
        print("Construyendo índice por primera vez...")
        embeddings, labels = load_embeddings()
        index.build(embeddings, labels)
    else:
        print("Índice cargado desde disco")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("No se pudo abrir la cámara")

    #FPS
    fps = 0
    frame_count = 0
    start_time = time.time()

    frame_id = 0

    #Memoria de resultados
    last_results = []
    last_update_time = 0
    RESULT_TTL = 1.0  # segundos

    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

        frame_id += 1

        #Tiempo total
        start_total = time.time()

        current_results = []

        #SOLO procesar algunos frames
        if frame_id % FRAME_SKIP == 0:

            faces = detect_faces(frame)

            for face in faces:

                embedding = generate_embedding(face)

                #Tiempo de búsqueda
                start_search = time.time()
                results = index.search(embedding)
                end_search = time.time()

                if len(results) == 0:
                    continue

                best_match = results[0]

                name = best_match["label"]
                score = best_match["score"]

                x1, y1, x2, y2 = face.bbox.astype(int)

                if score > THRESHOLD:
                    label = f"{name} ({score:.2f})"
                    color = (0, 255, 0)
                else:
                    label = "Desconocido"
                    color = (0, 0, 255)

                #Guardar resultado
                current_results.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": label,
                    "color": color
                })

                #Métrica
                search_time = end_search - start_search

            #actualizar memoria
            if current_results:
                last_results = current_results
                last_update_time = time.time()

        #Dibujar SIEMPRE usando memoria
        if time.time() - last_update_time < RESULT_TTL:

            for result in last_results:

                x1, y1, x2, y2 = result["bbox"]
                label = result["label"]
                color = result["color"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        #Tiempo total
        end_total = time.time()
        total_time = end_total - start_total

        try:
            print(f"Busqueda: {search_time:.4f}s | Total: {total_time:.4f}s")
        except:
            pass

        #FPS
        frame_count += 1

        if frame_count >= 10:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Reconocimiento Facial", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()