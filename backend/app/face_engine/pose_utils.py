import numpy as np


def check_pose(yaw, pose):

    if pose == "front":
        return abs(yaw) < 10

    if pose == "left":
        return yaw > 20

    if pose == "right":
        return yaw < -20

    return False


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