from camera_calibrator import CameraCalibrator
import yaml
import cv2
import numpy as np

if __name__ == "__main__":
    # Load sample transforms
    with open('data/transforms.yaml') as f:
        transforms_data = yaml.load(f, Loader=yaml.FullLoader)

    # Instantiate CameraCalibration class
    calib = CameraCalibrator(board_shape=(
        3, 4), tile_side=0.062, apriltag_families="tag36h10")

    # Prepare transforms, images data
    transforms = []
    images = []
    for i in range(2, 7):
        if i == 3 or i == 4:
            continue
        image = cv2.imread("data/" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
        trans = transforms_data["exp" + str(i)]["Translation"]
        quat = transforms_data["exp" + str(i)]["Quaternion"]
        transforms.append(calib.transquat_to_mat(trans, quat))
        images.append(image)

    # Get X (finetunning matrix)
    X = calib.eye_in_hand_finetunning(transforms, images)
    print(X)
