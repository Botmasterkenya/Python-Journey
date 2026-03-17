import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------------------------
# STEP 1: Create and save a test image
# ------------------------------------------

# Create a 200x200 orange image
# "RGB" = color image, (200, 200) = size, (255, 165, 0) = orange
image_pil = Image.new("RGB", (200, 200), (255, 165, 0))

# Save it to your project folder
image_pil.save("test_image.jpg")
print("Image saved successfully!")

# ------------------------------------------
# STEP 2: Open the image and convert to numbers
# ------------------------------------------

loaded_image = Image.open("test_image.jpg")

# Convert PIL image into a numpy array (grid of numbers)
image_array = np.array(loaded_image)

# ------------------------------------------
# STEP 3: Explore the numbers
# ------------------------------------------

print("Image shape:", image_array.shape)
print("Min pixel value:", image_array.min())
print("Max pixel value:", image_array.max())
print("Pixel at center:", image_array[100, 100])

# ------------------------------------------
# STEP 4: Draw a white square in the middle
# ------------------------------------------

# White = [255, 255, 255]
# We are editing rows 75-125, columns 75-125
image_array[75:125, 75:125] = [255, 255, 255]

# ------------------------------------------
# STEP 5: Display original and modified
# ------------------------------------------

plt.subplot(1, 2, 1)
plt.imshow(np.array(Image.open("test_image.jpg")))
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_array)
plt.title("Modified - White Square Added")
plt.axis('off')

plt.show()


