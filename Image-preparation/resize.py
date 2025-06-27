from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

input_dir = r'G:\My Drive\MasterThesisTest\CroppedData'
output_dir = r'G:\My Drive\MasterThesisTest\ResizedData2'  # Changed output directory name

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
                img = img.resize((512, 512), Image.Resampling.LANCZOS)

                # Replace all spaces in the original filename with underscores
                new_filename = filename.replace(" ", "_")
                resized_img_path = os.path.join(resized_species_path, new_filename)
                save_kwargs = {}
                if resized_img_path.lower().endswith(('.jpg','.jpeg')):
                    save_kwargs = {'quality':85, 'optimize':True}

                img.save(resized_img_path,format='WEBP', **save_kwargs)
                print(f"Resized and renamed: {img_path} -> {resized_img_path}")

                gray_orig = (Image.open(img_path)
                 .convert('L')
                 .resize((512,512), Image.LANCZOS))
                gray_res  = Image.open(resized_img_path).convert('L')
                score = ssim(np.array(gray_orig), np.array(gray_res))
                
                print("SSIM:", score)

print(f"\nResizing and renaming complete. Resized images are saved in: {output_dir}")