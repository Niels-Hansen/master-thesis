import os
import shutil
import random
import re
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class DataLoaderFactory:
    def __init__(self, source_dir, mapping_file, temp_dir):
        """
        source_dir   – root folder where subfolders are your raw IBT_* folders
        mapping_file – path to .xlsx that has columns ['IBT number','genus','species']
        temp_dir     – where to build the new class-structured tree
        """
        self.source_dir  = source_dir
        self.mapping_file = mapping_file
        self.temp_dir     = temp_dir
        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])
        # build IBT→(genus,species) lookup once
        df_map = pd.read_excel(self.mapping_file)

        # normalize code
        df_map['IBT_code'] = (
            df_map['IBT number']
            .astype(str)
            .str.replace(' ', '_', regex=False)   # "IBT 12085" → "IBT_12085"
        )
        # drop any extra rows with the same IBT_code
        df_map = df_map.drop_duplicates(subset='IBT_code', keep='first')

        # now it’s safe to set_index + to_dict
        self.lookup = df_map.set_index('IBT_code')[['genus','species']].to_dict(orient='index')

    def prepare_data(self, model_name):
        # clean slate
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(f"visualizations_{model_name}", exist_ok=True)

        random.seed(10)
        class_counts = {}

        # walk source_dir
        for folder in os.listdir(self.source_dir):
            folder_path = os.path.join(self.source_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for img in os.listdir(folder_path):
                if not img.lower().endswith(('.jpg','.jpeg','.png')):
                    continue

                m = re.search(r'(IBT_\d+)_\d+_([A-Za-z]+)_\d+min', img)
                if not m:
                    print(f"Skipping {img}: filename doesn’t match pattern.")
                    continue

                ibt_code = m.group(1)            # e.g. "IBT_23255"
                if ibt_code not in self.lookup:
                    print(f"No mapping for {ibt_code}, skipping {img}")
                    continue
                media    = m.group(2).upper()    # e.g. "MEA"

                # lookup genus/species
                if ibt_code not in self.lookup:
                    print(f"No mapping for {ibt_code}, skipping {img}")
                    continue
                genus   = self.lookup[ibt_code]['genus']
                species = self.lookup[ibt_code]['species']

                # new class folder
                class_name = f"{genus}_{species}_{media}"
                dest_dir   = os.path.join(self.temp_dir, class_name)
                os.makedirs(dest_dir, exist_ok=True)

                shutil.copy(
                    os.path.join(folder_path, img),
                    os.path.join(dest_dir, img)
                )
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # visualize counts
        plt.figure(figsize=(20, 20))
        items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"Items before zip: {items}")

        classes, counts = zip(*items)
        plt.bar(range(len(classes)), counts)
        plt.xticks(range(len(classes)), classes, rotation=90)
        plt.title("Class Distribution (genus_species_media)")
        plt.xlabel("Class")
        plt.ylabel("Number of Images")
        plt.tight_layout()
        plt.savefig(f"visualizations_{model_name}/class_distribution.png")
        plt.close()

        # return an ImageFolder on the new tree
          # Create dataset object
        dataset = datasets.ImageFolder(self.temp_dir, transform=self.data_transforms)
        dataset.classes = classes
        return dataset
