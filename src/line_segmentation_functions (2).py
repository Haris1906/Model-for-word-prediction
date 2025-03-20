import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def segment_words(image_path, output_dir="cropped_images"):
    """
    Segments words from an image, resizes them to 128x32, and saves them to the output directory.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory where segmented words will be saved.

    Returns:
        dict: Bounding box information for each cropped word.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Dilation to connect components
    kernel = np.ones((7, 15), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left to right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Initialize bounding boxes dictionary
    bounding_boxes = {}

    # Process each detected contour
    for i, ctr in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(ctr)
        filename = f"word_{i}.png"
        save_path = os.path.join(output_dir, filename)

        # Store bounding box info
        bounding_boxes[filename] = (x, y, w, h)

        # Crop the word region
        cropped_img = img[y:y+h, x:x+w]

        # Resize to 128x32
        

        # Display the resized cropped image
        plt.figure(figsize=(4, 4))
        plt.imshow(cropped_img, cmap='gray')
        plt.title(f"Cropped & Resized Image ({filename})")
        plt.axis("off")
        plt.show()

        # Save the resized image
        cv2.imwrite(save_path, cropped_img)

    print("Segmented words saved in:", output_dir)
    return bounding_boxes
