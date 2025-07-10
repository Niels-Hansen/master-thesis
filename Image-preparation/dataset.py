import os
import pandas as pd
import re
from collections import defaultdict

class DatasetSummary:
    def __init__(self, root_dir, mapping_file):
        self.root_dir = root_dir
        self.mapping_file = mapping_file

    def summarize(self, output_csv="dataset_summary.csv"):
        df = pd.read_excel(self.mapping_file)
        df['IBT_code'] = df['IBT number'].astype(str).str.replace(' ', '_', regex=False)

        df['genus'] = df['genus'].astype(str).str.strip().str.replace('"', '')
        df['species'] = df['species'].astype(str).str.strip().str.replace('"', '')

        ibt_to_class = df.set_index('IBT_code').apply(
            lambda row: (
                f"{str(row['genus']).strip()}" if pd.isna(row['species']) or str(row['species']).strip().lower() in ['nan', '']
                else f"{str(row['genus']).strip()}_{str(row['species']).strip()}"
            ) if not pd.isna(row['genus']) and str(row['genus']).strip().lower() not in ['nan', '']
            else '',
            axis=1
        ).to_dict()

        class_image_counts = defaultdict(int)
        unmatched_folders = []
        unmatched_images = []
        total_images = 0

        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path) or not folder.startswith("IBT_"):
                continue

            if folder not in ibt_to_class:
                unmatched_folders.append(folder)
                continue

            class_name = ibt_to_class[folder]
            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    class_image_counts[class_name] += 1
                    total_images += 1
                else:
                    unmatched_images.append(os.path.join(folder, filename))

        df_summary = pd.DataFrame([
            {"Class": cls, "Image Count": count}
            for cls, count in sorted(class_image_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        df_summary.to_csv(output_csv, index=False)

        print("Dataset Summary for Report")
        print(f"Total images: {total_images}")
        print(f"Total classes (genus + species): {len(class_image_counts)}")
        print(f"Saved class distribution to: {output_csv}")

        if unmatched_folders:
            print(f"\n Unmatched IBT folders (not in Excel):")
            for f in unmatched_folders:
                print(f"  {f}")

        return {
            "total_images": total_images,
            "num_classes": len(class_image_counts),
            "unmatched_folders": unmatched_folders,
            "unmatched_images": unmatched_images,
            "summary_file": output_csv
        }
        
summary = DatasetSummary(
root_dir=r"G:\My Drive\MasterThesis\CircularMaskedDataSorted\300min",
mapping_file=r"G:\My Drive\MasterThesis\imageAnalysis_info.xlsx"
)
stats = summary.summarize(output_csv="class_summary300.csv")