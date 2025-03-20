import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def segment_words(image_path, output_dir="cropped_images"):
    """
    Segments words from an image, increases width by 50% and height by 20%, and saves them to the output directory.

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
        cropped_img = img[y:y + h, x:x + w]

        # Get dimensions correctly
        h, w = cropped_img.shape[:2]

        # Increase width by 50% and height by 20%
        new_pani_width = int(w + 30)
        new_pani_height = int(h + 15)

        # Create a new uniform background with the same color as top-left pixel
        background_color = cropped_img[0, 0]  # Get color from (0, 0) as tuple (B, G, R)
        uniform_background = np.full((new_pani_height, new_pani_width, 3), background_color, dtype=np.uint8)

        # Calculate position to paste the original image in the center
        pani_position_x = (new_pani_width - w) // 2
        pani_position_y = (new_pani_height - h) // 2

        # Paste the original cropped image onto the uniform background
        uniform_background[pani_position_y:pani_position_y + h, pani_position_x:pani_position_x + w] = cropped_img

        # Display the resized cropped image
        plt.figure(figsize=(4, 4))
        plt.imshow(cv2.cvtColor(uniform_background, cv2.COLOR_BGR2RGB))
        plt.title(f"Cropped & Resized Image ({filename})")
        plt.axis("off")
        plt.show()

        # Save the resized image
        cv2.imwrite(save_path, uniform_background)

    print("Segmented words saved in:", output_dir)
    return bounding_boxes

