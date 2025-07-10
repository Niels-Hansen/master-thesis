import os
import random
import re
import shutil
import pandas as pd
import glob
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report


class DataSplitter:

    def __init__(self, source_dir, mapping_file, output_dir):
        self.source_dir = source_dir
        self.mapping_file = mapping_file
        self.output_dir = output_dir
        self.train_dir = os.path.join(output_dir, 'train')
        self.val_dir = os.path.join(output_dir, 'val')
        self.logs_dir   = os.path.join('media', 'basic_validation', 'yolo11n_fungi_val')

        df = pd.read_excel(self.mapping_file)
        df['IBT_code'] = df['IBT number'].astype(str).str.replace(' ', '_', regex=False)

        df['genus'] = df['genus'].astype(str).str.strip().str.replace('"', '')
        df['species'] = df['species'].astype(str).str.strip().str.replace('"', '')

        self.lookup = df.set_index('IBT_code').apply(
            lambda row: (
                f"{str(row['genus']).strip()}" if pd.isna(row['species']) or str(row['species']).strip().lower() in ['nan', '']
                else f"{str(row['genus']).strip()}_{str(row['species']).strip()}"
            ) if not pd.isna(row['genus']) and str(row['genus']).strip().lower() not in ['nan', '']
            else '',
            axis=1
        ).to_dict()
                
    def prepare_split(self, split_ratio=0.8):
        
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

        for folder in os.listdir(self.source_dir):
            folder_path = os.path.join(self.source_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            
            for img_name in os.listdir(folder_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                m = re.match(r"(IBT_\d+)_\d+_([A-Za-z]+)_\d+min", img_name)
                if not m:
                    continue
                
                ibt, media = m.group(1), m.group(2).upper()
                if ibt not in self.lookup:
                    continue
                
                class_name = self.lookup[ibt]

                destination_base = self.train_dir if random.random() < split_ratio else self.val_dir
                
                destination_class_dir = os.path.join(destination_base, class_name)
                os.makedirs(destination_class_dir, exist_ok=True)

                source_file_path = os.path.join(folder_path, img_name)
                destination_file_path = os.path.join(destination_class_dir, img_name)
                shutil.copy(source_file_path, destination_file_path)

        print("Data preparation complete")
   


if __name__ == '__main__':

    source_images_dir = "/work3/s233780/CircularMaskedData"
    #source_images_dir = r"G:\My Drive\MasterThesisTest\CircularMaskedData"
    mapping_file = "imageAnalysis_info.xlsx"
    output_dataset_dir = "/work3/s233780/fungi_dataset"

    try:
        splitter = DataSplitter(source_images_dir, mapping_file, output_dataset_dir)
        splitter.prepare_split(split_ratio=0.8) 
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        exit()

    try:

        model = YOLO('yolo11n-cls.pt')

        print("Starting training")
        results = model.train(
            data=output_dataset_dir,
            pretrained=True,
            epochs=10, 
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
            save_json=True,
            save_period=2,
            device='0', # 0 for GPU, 'cpu' for CPU
        )
        print("Training complete.")

        res = results[0] if isinstance(results, list) else results
        epoch = getattr(res, 'epoch', getattr(res, 'epochs', None))
        train_metrics = getattr(res, 'results_dict', {})
        print(f"Training metrics: {train_metrics}")
        print("Running final validation…")
        valm = model.val(
                         imgsz=224,
                         batch=128,
                         plots=True,
                         name='yolo11n_fungi_val',
                         project='validation/basic_validation',
                         save=True,
                         save_json=True,
                         save_conf=True,
                         device='0',
                         )
        val_dir = os.path.join(output_dataset_dir, 'val')
        val_paths = glob.glob(os.path.join(val_dir, '*', '*.[jp][pn]g'), recursive=True)
        val_top1    = getattr(valm, 'top1', None)
        val_top5    = getattr(valm, 'top5', None)
        val_fitness = getattr(valm, 'fitness', None)
        def get_media_from_path(path: str) -> str:
            stem = os.path.splitext(os.path.basename(path))[0]
            m = re.match(r"(IBT_\d+)_\d+_([A-Za-z]+)_\d+min", stem)
            return m.group(2).upper() if m else "UNKNOWN"
        results = model.predict(val_paths, imgsz=224, batch=128, device='0', verbose=False, stream=True)
        y_pred = [int(r.probs.top1) for r in results]
        idx2name = model.names
        name2idx = {n: i for i, n in idx2name.items()}
        y_true_cls = [os.path.basename(os.path.dirname(p)) for p in val_paths]
        y_true = [name2idx[c] for c in y_true_cls]
        
        df_media = pd.DataFrame({
            'path':      val_paths,
            'true':      y_true,
            'pred':      y_pred,
            'media':     [get_media_from_path(p) for p in val_paths]
        })
        df_media['true_class'] = df_media['true'].map(idx2name)
        df_media['pred_class'] = df_media['pred'].map(idx2name)

        logs_dir = os.path.join('media', 'yolo11n_fungi_val')
        os.makedirs(logs_dir, exist_ok=True)
        fn_img = os.path.join(logs_dir, f"yolo_fold_epoch{epoch}_media_predictions.csv")
        df_media.to_csv(fn_img, index=False)
        
        y_true2 = df_media['true'].tolist()
        y_pred2 = df_media['pred'].tolist()
        
        
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        report = classification_report(
            y_true, y_pred,
            target_names=[idx2name[i] for i in sorted(idx2name)],
            zero_division=0,
            output_dict=True
        )
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(os.path.join(logs_dir, f"classification_report_epoch{epoch}.csv"))
        print(df_report)
                
        
        summary = []
        for media, sub in df_media.groupby('media'):
            acc = accuracy_score(sub['true'], sub['pred'])
            m_f1 = f1_score(sub['true'], sub['pred'], average='macro', zero_division=0)
            summary.append({
                'media':     media,
                'n_samples': len(sub),
                'accuracy':  acc,
                'macro_f1':  m_f1
            })
            
        media_true_counts = df_media.pivot_table(
            index='media',
            columns='true_class',
            values='path',
            aggfunc='count',
            fill_value=0
        )

        media_pred_counts = df_media.pivot_table(
            index='media',
            columns='pred_class',
            values='path',
            aggfunc='count',
            fill_value=0
        )    
            
        df_media['correct'] = (df_media['true_class'] == df_media['pred_class']).astype(int)
        media_acc = df_media.groupby(['media','true_class'])['correct'] \
                            .mean() \
                            .unstack(fill_value=0) \
                            .rename_axis(columns=None)    
        summary_df = pd.DataFrame(summary).sort_values('accuracy', ascending=False)

        fn_sum = os.path.join(logs_dir, f"yolo_epoch{epoch}_media_summary.csv")
        summary_df.to_csv(fn_sum, index=False)
        media_true_counts.to_csv(f"{logs_dir}/media_true_counts.csv")
        media_pred_counts.to_csv(f"{logs_dir}/media_pred_counts.csv")
        media_acc.to_csv(f"{logs_dir}/media_class_accuracy.csv")

        print(f"Saved media summary → {fn_sum}")
        row = {
            'timestamp': pd.Timestamp.now(),
            'epoch':     epoch,
            'train_top1':       train_metrics.get('metrics/accuracy_top1'),
            'train_top5':       train_metrics.get('metrics/accuracy_top5'),
            'train_fitness':    train_metrics.get('fitness'),
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

        
    except Exception as e:
        print(f"An error occurred during model training or validation: {e}")