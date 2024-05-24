from ultralytics import YOLO
from PIL import Image
import os

input_directory = "cv_advanced_input"
output_directory = "cv_advanced_output"

model = YOLO("yolov8s.pt")

if not os.path.exists(output_directory): os.makedirs(output_directory)

for file in os.listdir(input_directory):
    if file.endswith(".jpg"):
        results = model.predict(os.path.join(input_directory, file))
        result_image = Image.fromarray(results[0].plot()[:,:,::-1])
        result_image.save(os.path.join(output_directory, file))


