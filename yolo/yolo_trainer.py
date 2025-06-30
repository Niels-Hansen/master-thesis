import os
import random
import re
import shutil
import pandas as pd
import wandb
from ultralytics import YOLO

class DataSplitter:

    def __init__(self, source_dir, mapping_file, output_dir):
        self.source_dir = source_dir
        self.mapping_file = mapping_file
        self.output_dir = output_dir
        self.train_dir = os.path.join(output_dir, 'train')
        self.val_dir = os.path.join(output_dir, 'val')

        # Load the mapping from IBT code to genus/species
        if not os.path.isfile(self.mapping_file):
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_file}")
        df_map = pd.read_excel(self.mapping_file)
        df_map['IBT_code'] = (
            df_map['IBT number'].astype(str)
                           .str.replace(' ', '_', regex=False)
        )
        df_map = df_map.drop_duplicates(subset='IBT_code', keep='first')
        self.lookup = df_map.set_index('IBT_code')[['genus','species']].to_dict(orient='index')

    def prepare_split(self, split_ratio=0.8):
        print(f"Preparing data split in '{self.output_dir}'...")

        # Clean existing output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        # Create train and val directories
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

        # Iterate through all source images and copy them to train or val
        for folder in os.listdir(self.source_dir):
            folder_path = os.path.join(self.source_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            
            for img_name in os.listdir(folder_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                # Extract class name from the image file name
                m = re.match(r"(IBT_\d+)_\d+_([A-Za-z]+)_\d+min", img_name)
                if not m:
                    continue
                
                ibt, media = m.group(1), m.group(2).upper()
                if ibt not in self.lookup:
                    continue
                
                genus = self.lookup[ibt]['genus']
                species = self.lookup[ibt]['species']
                class_name = f"{genus}_{species}"

                # Decide if this image goes to train or val
                destination_base = self.train_dir if random.random() < split_ratio else self.val_dir
                
                # Create the class directory inside train/val
                destination_class_dir = os.path.join(destination_base, class_name)
                os.makedirs(destination_class_dir, exist_ok=True)

                # Copy the file
                source_file_path = os.path.join(folder_path, img_name)
                destination_file_path = os.path.join(destination_class_dir, img_name)
                shutil.copy(source_file_path, destination_file_path)

        print("Data preparation complete.")
        print(f"Train data: {self.train_dir}")
        print(f"Validation data: {self.val_dir}")


if __name__ == '__main__':

    source_images_dir = "/work3/s233780/CircularMaskedData"
    #source_images_dir = r"G:\My Drive\MasterThesisTest\ResizedData"
    mapping_file = "imageAnalysis_info.xlsx"
    output_dataset_dir = "../fungi_dataset"

    try:
        splitter = DataSplitter(source_images_dir, mapping_file, output_dataset_dir)
        splitter.prepare_split(split_ratio=0.8) 
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        exit()

    try:

        model = YOLO('yolo11n-cls.pt')

        print("\nStarting model training...")
        results = model.train(
            data=output_dataset_dir,
            pretrained=True,
            epochs=20, 
            imgsz=224,
            batch=128,
            lr0=0.001, 
            flipud=0.5,
            fliplr=0.5,
            degrees=20.0,
            hsv_v=0.2,
            hsv_s=0.2,
            hsv_h=0.1,
            optimizer='adam',
            project='runs/basic_training',
            name='yolo11n_fungi',
            plots=True,
            save=True,
            save_period=2,
            device='0', # 0 for GPU, 'cpu' for CPU
        )
        print("Model training complete.")

        res = results[0] if isinstance(results, list) else results
        epoch = getattr(res, 'epoch', getattr(res, 'epochs', None))
        train_metrics = getattr(res, 'results_dict', {})
        print(f"Training metrics: {train_metrics}")
        print("\nRunning final validation…")
        valm = model.val(save_json=True,
                         imgsz=224,
                         batch=64,
                         plots=True,
                         name='yolo11n_fungi_val',
                         project='runs/basic_validation',
                         save_txt=True,
                         save_conf=True,
                         )
        
        val_top1    = getattr(valm, 'top1', None)
        val_top5    = getattr(valm, 'top5', None)
        val_fitness = getattr(valm, 'fitness', None)
        row = {
            'timestamp': pd.Timestamp.now(),
            'epoch':     epoch,
            # training
            'train_top1':       train_metrics.get('metrics/accuracy_top1'),
            'train_top5':       train_metrics.get('metrics/accuracy_top5'),
            'train_fitness':    train_metrics.get('fitness'),
            # validations
            'val_top1':         val_top1,
            'val_top5':         val_top5,
            'val_fitness':      val_fitness,       
        }
        out = "train_metrics.csv"
        pd.DataFrame([row]).to_csv(
            out,
            index=False,
            mode='a',
            header=not os.path.exists(out)
        )
        print(f"Final validation Top-1 = {valm.top1}")        
        config = {
            'epochs':20, 'imgsz':224, 'batch':64,
            'train_dir': output_dataset_dir
        }
        wandb.init(project="yolo-trials", config=config)
        wandb.log({'train_epoch': epoch, **train_metrics,
                   'val_top1': getattr(valm, 'top1', None)})
        
    except Exception as e:
        print(f"An error occurred during model training or validation: {e}")