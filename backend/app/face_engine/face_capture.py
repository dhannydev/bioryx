import cv2
import numpy as np
import os

from .embedding_generator import (
    detect_faces,
    generate_embedding
)

from .face_quality import (
    is_blurry,
    face_too_small
)

from .pose_utils import (
    check_pose,
    is_stable
)

# Carpeta de embeddings
os.makedirs("embeddings", exist_ok=True)


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

            faces = detect_faces(frame)

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

            if captured_poses[pose_name]:
                captured = True
                continue

            embedding = generate_embedding(face)

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