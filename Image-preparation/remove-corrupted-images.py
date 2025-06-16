from PIL import Image
import numpy as np
import os

def is_corrupted(image_path, threshold=5):
    try:
        img = Image.open(image_path).convert('L')  
        return np.mean(np.array(img)) < threshold
    except:
        return True  

def clean_folder(root_dir, log_path="bad_images_log.txt"):
    bad_images = []
    for ibt_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, ibt_folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if is_corrupted(img_path):
                bad_images.append(img_path)
                os.remove(img_path)


    with open(log_path, "w") as f:
        for img_path in bad_images:
            f.write(f"{img_path}\n")
    print(f"Removed {len(bad_images)} bad images. Log saved to {log_path}")

clean_folder(r"G:\My Drive\MasterThesis\ResizedDataFixed")
