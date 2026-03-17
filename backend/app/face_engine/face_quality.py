import cv2

def is_blurry(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return variance < 80


def face_too_small(face):

    x1, y1, x2, y2 = face.bbox.astype(int)

    width = x2 - x1
    height = y2 - y1

    return width < 120 or height < 120