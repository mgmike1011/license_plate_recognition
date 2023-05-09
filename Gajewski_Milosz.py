import argparse
import json
from pathlib import Path

import cv2

from processing.utils import perform_processing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    # images_dir = '/home/milosz/RiSA_1/SW/train' #Debug
    results_file = Path(args.results_file)
    # results_file = '/home/milosz/RiSA_1/SW/output.txt' #Debug
    # python3 Gajewski_Milosz.py /home/milosz/RiSA_1/SW/train /home/milosz/RiSA_1/SW/output.txt

    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    results = {}
    for image_path in images_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue

        results[image_path.name] = perform_processing(image)

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()
