import openpifpaf
from PIL import Image
import matplotlib.pyplot as plt

# Step 3: Create Predictor
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16-apollo-24')

# Step 4: Load Image
image_path = 'screenshot2.png'  # Provide the path to your image file
image = Image.open(image_path)

# Step 5: Run Prediction
predictions, gt_anns, image_meta = predictor.pil_image(image)

# Step 6: Process Results
# Plot the image with keypoints
fig, ax = plt.subplots()
openpifpaf.show.KeypointPainter(show_box=False, color_connections=True).draw(ax, predictions)
ax.imshow(image)
plt.show()