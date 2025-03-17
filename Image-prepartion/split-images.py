from PIL import Image, ImageDraw, ImageFont
import os
import re

input_folder = r"C:\Users\niels\Documents\04_Semester\Data"  
output_folder = r"C:\Users\niels\Documents\04_Semester\CroppedData"
os.makedirs(output_folder, exist_ok=True)

media_types = ["CREA", "PDA", "YES", "CYA", "MEAox", "OAT"]


image_files = sorted(
    [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpeg'))]
)

images_per_day = 48
last_2_days_images = image_files[-(2 * images_per_day):] 


total_selected_images = len(last_2_days_images)
time_intervals = 30  
estimated_timestamps = [(i * time_intervals) for i in range(total_selected_images)]

try:
    font = ImageFont.truetype("arial.ttf", 40)
except IOError:
    font = ImageFont.load_default()

for i, image_file in enumerate(last_2_days_images):
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path)

    original_name, ext = os.path.splitext(image_file)

    timestamp = estimated_timestamps[i]

    width, height = image.size
    rows, cols = 3, 2  
    square_width = width // cols
    square_height = height // rows

    for row in range(rows):
        for col in range(cols):
            left = col * square_width
            upper = row * square_height
            right = left + square_width
            lower = upper + square_height

            cropped_image = image.crop((left, upper, right, lower))

            media = media_types[row * cols + col]

            label_text = f"{media} {timestamp} min"

            draw = ImageDraw.Draw(cropped_image)
            text_position = (10, 10)
            draw.text(text_position, label_text, fill="white", font=font)

            new_filename = f"{original_name}_{media}_{timestamp}min{ext}"
            cropped_image.save(os.path.join(output_folder, new_filename))

print(f"Labeled images from the last 2 days saved in {output_folder}")
