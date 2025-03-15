import argparse
import os
import time
import cv2
from Parallelized_seam_carving import SeamCarver

def main():
    parser = argparse.ArgumentParser(
        description="Seam carving image resizing tool. "
                    "Provide an input image and desired output dimensions."
    )
    parser.add_argument(
        '-i', '--input',
        default='in/images/image.jpg',
        help="Path to the input image file. Defaults to in/images/image.jpg."
    )
    parser.add_argument(
        '-o', '--output',
        default='out/images/image_result.png',
        help="Path to save the output image. Defaults to out/images/image_result.png."
    )
    parser.add_argument(
        '-H', '--height',
        type=int,
        default=None,
        help="New height for the output image. If not provided, defaults to the original image height."
    )
    parser.add_argument(
        '-W', '--width',
        type=int,
        default=None,
        help="New width for the output image. If not provided, defaults to the original image width."
    )
    args = parser.parse_args()

    input_image = args.input
    output_image = args.output

    # Load image to obtain original dimensions if height or width are not provided
    image = cv2.imread(input_image)
    if image is None:
        print(f"Error: Cannot read image from {input_image}")
        return
    orig_height, orig_width = image.shape[:2]

    new_height = args.height if args.height is not None else orig_height
    new_width = args.width if args.width is not None else orig_width

    print(f"Resizing image {input_image} to {new_width}x{new_height}...")
    start_time = time.time()

    obj = SeamCarver(input_image, new_height, new_width)
    obj.save_result(output_image)

    end_time = time.time()
    print(f"Image resize took {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    main()
