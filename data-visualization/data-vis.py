import os

def count_ibt_folders_and_images(base_path):
    ibt_folders_count = 0
    total_images_count = 0

    for root, dirs, files in os.walk(base_path):
        if 'IBT' in root:
            ibt_folders_count += 1
            images_count = len([file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
            total_images_count += images_count
            print(f"Folder: {root}, Images: {images_count}")

    print(f"\nTotal IBT folders: {ibt_folders_count}")
    print(f"Total images in IBT folders: {total_images_count}")

if __name__ == "__main__":
    base_path = r"G:\My Drive\MasterThesis\Data"
    if os.path.isdir(base_path):
        count_ibt_folders_and_images(base_path)
    else:
        print("Invalid directory path.")