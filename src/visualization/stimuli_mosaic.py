from PIL import Image
import os
import random
import numpy as np

# Set paths and parameters
image_folder = "figures/stimuli/original"
output_image_path = "figures/stimuli/mosaic.png"
bg = np.array([128, 128, 128, 255], dtype=np.int64)  # Background color for the collage
canvas_size = (1920, 1080)  # Size of the entire collage canvas
image_size = (150, 150)  # Desired size of each image
overlap = 0.25  # Overlap between images
num_images = int((canvas_size[0]//(image_size[0]*(1 - overlap))) \
             * (canvas_size[1]//(image_size[1]*(1 - overlap)))) * 2

# Load images, resize them, and remove background (assuming a background color of #767676)
images = []
for file in os.listdir(image_folder):
    if file.endswith(('tif', 'tiff')):
        img = Image.open(os.path.join(image_folder, file)).resize(image_size).convert('RGBA')
        img = np.array(img)

        img[(img == bg).all(axis=2)] = np.array([255, 255, 255, 0], dtype=np.int64) 
        img = Image.fromarray(img)
        images.append(img)

# Create a blank canvas for the collage
collage = Image.new('RGBA', canvas_size, (255, 255, 255, 0))

x_pos = np.linspace(0, canvas_size[0], int(canvas_size[0]//((1 - overlap)*image_size[0])), endpoint=True).tolist()
y_pos = np.linspace(0, canvas_size[1], int(canvas_size[1]//((1 - overlap)*image_size[1])), endpoint=True).tolist()
spaces = np.array(np.meshgrid(x_pos, y_pos)).T.reshape(-1, 2)

for idx in range(num_images):
    pos = spaces[idx%len(spaces)]
    img = random.choice(images)

    if idx >= len(spaces):
        pos = (pos[0] + (image_size[0]*0.75)//2, pos[1] + (image_size[1]*0.75)//2)
    # Paste the image, allowing overlap
    collage.paste(img, (int(pos[0].item()), int(pos[1].item())), mask=img)

# Save the final collage
collage = collage.convert('RGB')  # Convert back to RGB for saving
collage.save(output_image_path, format='PNG')


