from PIL import Image
import os

input_dir = r'G:\My Drive\MasterThesis\CroppedData'
output_dir = r'G:\My Drive\MasterThesis\ResizedData'  # Changed output directory name

os.makedirs(output_dir, exist_ok=True)

for species_dir in os.listdir(input_dir):
    species_path = os.path.join(input_dir, species_dir)
    if os.path.isdir(species_path):
        # Replace spaces in the species directory name for the output directory
        resized_species_dirname = species_dir.replace(" ", "_")
        resized_species_path = os.path.join(output_dir, resized_species_dirname)
        os.makedirs(resized_species_path, exist_ok=True)

        for filename in os.listdir(species_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(species_path, filename)
                img = Image.open(img_path)
                img = img.resize((512, 512))

                # Replace all spaces in the original filename with underscores
                new_filename = filename.replace(" ", "_")
                resized_img_path = os.path.join(resized_species_path, new_filename)

                img.save(resized_img_path)
                print(f"Resized and renamed: {img_path} -> {resized_img_path}")

print(f"\nResizing and renaming complete. Resized images are saved in: {output_dir}")