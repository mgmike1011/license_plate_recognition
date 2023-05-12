import numpy as np
import cv2


def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # lokalziacja tablicy
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # TODO: add image processing here
    return 'PO12345'