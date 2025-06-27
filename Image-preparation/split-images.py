from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import sys
print (sys.path)
from natsort import natsorted 

# Input and output folders
input_base_folder = r"G:\My Drive\MasterThesis\Data"  
output_base_folder = r"G:\My Drive\MasterThesis\CroppedData"
os.makedirs(output_base_folder, exist_ok=True)

media_types = ["CREA", "PDA", "YES", "CYA", "MEAox", "OAT"]

def process_folder(input_folder):
    relative_path = os.path.relpath(input_folder, input_base_folder)
    ibt_number = relative_path.split(os.sep)[0]

    output_folder = os.path.join(output_base_folder, ibt_number)
    os.makedirs(output_folder, exist_ok=True)

    image_files = natsorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith('.jpeg')]
    )

    images_per_day = 48
    if len(image_files) < 2*images_per_day:
        print(f"Skipping folder {input_folder}, not enough images.")
        return

    last_2_days_images = image_files[-(2 * images_per_day):]

    for i, image_file in enumerate(last_2_days_images):
        image_path = os.path.join(input_folder, image_file)
        try:
   
            image = Image.open(image_path)
            image.verify
            image = Image.open(image_path)
            image.load()
            original_name, ext = os.path.splitext(image_file)

            timestamp = i * 30

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

                    new_filename = f"{ibt_number}_{original_name}_{media}_{timestamp}min{ext}"
                    cropped_image.save(os.path.join(output_folder, new_filename))
        except (OSError, IOError):
            print(f"Corrupt or unreadable image detected: {image_path}")

    print(f"Processed folder: {input_folder} -> Saved in: {output_folder}")

for root, dirs, files in os.walk(input_base_folder):
    if any(file.lower().endswith('.jpeg') for file in files):
        process_folder(root)

print("All folders processed successfully!")
