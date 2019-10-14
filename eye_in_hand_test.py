from camera_calibrator import CameraCalibrator

if __name__ == "__main__":
    # Load sample transforms
    with open('data/transforms.yaml') as f:
        transforms_data = yaml.load(f, Loader=yaml.FullLoader)
    
    # Instantiate CameraCalibration class
    calib = CameraCalibrator(board_shape=(3, 4), tile_side=0.062, apriltag_families="tag36h10")

    # Prepare transforms, images data
    transforms = []
    images = []
    for i in range(1, 7):
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

    Test apriltag
    calib = CameraCalibrator(board_shape=(3, 4), tile_side=0.062, apriltag_families="tag36h10")
    image = cv2.imread("data/new.png", cv2.IMREAD_GRAYSCALE)
    print(calib.get_apriltag_center(image))