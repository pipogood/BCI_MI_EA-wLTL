import keras_ocr
from PIL import Image, ImageDraw

# Initialize the Keras-OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Load the image
image_path = 'pic1_BSTM.png'
image = keras_ocr.tools.read(image_path)

# Perform OCR using Keras
prediction_groups = pipeline.recognize([image])

# Convert Keras-OCR result back to PIL Image for drawing
pil_image = Image.fromarray(image)

# Create a drawing object to edit the image
draw = ImageDraw.Draw(pil_image)

# Extract positions of each word and draw circles at word centers
centers = []
for prediction in prediction_groups[0]:  # Loop over the first image (there's only one in this case)
    word = prediction[0]
    box = prediction[1]
    
    # Calculate the center of the bounding box
    center_x = int((box[0][0] + box[2][0]) / 2)
    center_y = int((box[0][1] + box[2][1]) / 2)
    
    centers.append((word, (center_x, center_y)))

    # Draw a small circle at the center of the word
    radius = 5
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill='red', outline='black')

# Print the word centers
for word, center in centers:
    print(f"Word: {word}    ,   Center: {center}")

# Save or display the image with the centers marked
pil_image.show()  # Opens the image with default image viewer
# pil_image.save('output_image_with_centers.jpg')  # Uncomment to save the result
