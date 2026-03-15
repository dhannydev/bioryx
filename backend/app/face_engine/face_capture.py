import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os

# Inicializar InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Carpeta para guardar embeddings
os.makedirs("embeddings", exist_ok=True)


# -------------------------------
# Verificar si la imagen es borrosa
# -------------------------------
def is_blurry(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return variance < 80


# -------------------------------
# Verificar si el rostro está muy lejos
# -------------------------------
def face_too_small(face):

    x1, y1, x2, y2 = face.bbox.astype(int)

    width = x2 - x1
    height = y2 - y1

    return width < 120 or height < 120


# -------------------------------
# Verificar pose
# -------------------------------
def check_pose(yaw, pose):

    if pose == "front":
        return abs(yaw) < 10

    if pose == "left":
        return yaw > 20

    if pose == "right":
        return yaw < -20

    return False


# -------------------------------
# Verificar estabilidad de cabeza
# -------------------------------
def is_stable(face, last_bbox, last_yaw):

    x1, y1, x2, y2 = face.bbox.astype(int)
    pitch, yaw, roll = face.pose

    current_bbox = np.array([x1, y1, x2, y2])

    if last_bbox is None:
        return False, current_bbox, yaw

    bbox_diff = np.linalg.norm(current_bbox - last_bbox)
    yaw_diff = abs(yaw - last_yaw)

    if bbox_diff < 10 and yaw_diff < 3:
        return True, current_bbox, yaw

    return False, current_bbox, yaw


# -------------------------------
# Función principal de registro
# -------------------------------
def capture_face_embeddings(user_id):

    cap = cv2.VideoCapture(0)

    poses = [
        ("Mire al frente", "front"),
        ("Gire a la izquierda", "left"),
        ("Gire a la derecha", "right")
    ]

    embeddings = []

    captured_poses = {
        "front": False,
        "left": False,
        "right": False
    }

    stable_frames = 0
    required_stable_frames = 15

    last_bbox = None
    last_yaw = None

    for instruction, pose_name in poses:

        captured = False

        while not captured:

            ret, frame = cap.read()
            if not ret:
                continue

            faces = app.get(frame)

            if len(faces) != 1:

                cv2.putText(
                    frame,
                    "Debe haber solo un rostro",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.imshow("Registro Facial", frame)
                cv2.waitKey(1)
                continue

            face = faces[0]

            pitch, yaw, roll = face.pose

            x1, y1, x2, y2 = face.bbox.astype(int)

            face_crop = frame[y1:y2, x1:x2]

            # Verificar distancia
            if face_too_small(face):

                cv2.putText(
                    frame,
                    "Acérquese a la cámara",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.imshow("Registro Facial", frame)
                cv2.waitKey(1)
                continue

            # Verificar borrosidad
            if is_blurry(face_crop):

                cv2.putText(
                    frame,
                    "Imagen borrosa",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.imshow("Registro Facial", frame)
                cv2.waitKey(1)
                continue

            # Verificar pose correcta
            if not check_pose(yaw, pose_name):

                cv2.putText(
                    frame,
                    instruction,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                cv2.imshow("Registro Facial", frame)
                cv2.waitKey(1)
                continue

            # Verificar estabilidad
            stable, last_bbox, last_yaw = is_stable(face, last_bbox, last_yaw)

            if stable:
                stable_frames += 1
            else:
                stable_frames = 0

            if stable_frames < required_stable_frames:

                cv2.putText(
                    frame,
                    "Mantenga la cabeza quieta",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.imshow("Registro Facial", frame)
                cv2.waitKey(1)
                continue

            # Evitar capturar pose duplicada
            if captured_poses[pose_name]:
                captured = True
                continue

            # Capturar embedding
            embedding = face.embedding
            embeddings.append(embedding)

            captured_poses[pose_name] = True

            cv2.putText(
                frame,
                f"Capturado: {pose_name}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Registro Facial", frame)
            cv2.waitKey(1000)

            stable_frames = 0
            last_bbox = None
            last_yaw = None

            captured = True

    cap.release()
    cv2.destroyAllWindows()

    embeddings = np.array(embeddings)

    np.save(f"embeddings/{user_id}.npy", embeddings)

    print("Embeddings guardados correctamente")