import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from utils import Utils

class Evaluator:
    """
    Evaluator handles model inference, metric computation, report saving,
    and delegates visualization to VisualizationUtils.
    """
    def __init__(self, model, device, criterion, class_names, utils, logs_dir="logs"):
        self.model       = model
        self.device      = device
        self.criterion   = criterion
        self.class_names = class_names
        self.utils         = utils
        self.logs_dir    = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        
    @staticmethod
    def get_media_from_path(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0]
        return stem.split('_')[-2] 
    
    def evaluate(self, loader, model_name="model", fold=1, epoch=1):
        # Switch to eval mode
        self.model.eval()
        
        ds = loader.dataset
        if hasattr(ds, "dataset") and hasattr(ds, "indices"):
            base, idx = ds.dataset, ds.indices
        else:
            base, idx = ds, range(len(ds.samples))
        all_paths = [base.samples[i][0] for i in idx]

        all_preds = []
        all_labels = []
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                total_loss += loss.item() * labels.size(0)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
        df_media = pd.DataFrame({
            'true': all_labels,
            'pred': all_preds,
            'media': [self.get_media_from_path(p) for p in all_paths]
        })
        df_media['true_class'] = df_media['true'].apply(lambda x: self.class_names[x])
        df_media['pred_class'] = df_media['pred'].apply(lambda x: self.class_names[x])
        df_media.to_csv(
            os.path.join(self.logs_dir, f"{model_name}_fold{fold}_epoch{epoch}_media_predictions.csv"),
            index=False
        )
         # build media‐level summary
        summary = []
        for media, sub in df_media.groupby('media'):
            acc  = accuracy_score(sub['true'], sub['pred'])
            m_f1 = f1_score(sub['true'], sub['pred'], average='macro')
            summary.append({
                'media': media,
                'n_samples': len(sub),
                'accuracy': acc,
                'macro_f1': m_f1
            })
            print(f"--- {media} ({len(sub)} samples) ---")
            print(f"Accuracy: {acc:.4f}, Macro-F1: {m_f1:.4f}\n")

        # save media summary to CSV
        summary_df = pd.DataFrame(summary).sort_values('accuracy', ascending=False)
        media_csv = os.path.join(
            self.logs_dir,
            f"{model_name}_fold{fold}_epoch{epoch}_media_summary.csv"
        )
        summary_df.to_csv(media_csv, index=False)
        print(f"Saved media summary to {media_csv}")
            
        # Compute averaged metrics
        avg_val_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # Classification report 
        report_dict = classification_report(
            all_labels,
            all_preds,
            labels=range(len(self.class_names)),
            target_names=self.class_names,
            zero_division=0,
            output_dict=True
        )
        # Save report to CSV
        df_report = pd.DataFrame(report_dict).transpose()
        report_fname = f"{model_name}_fold{fold}_epoch{epoch}_report.csv"
        df_report.to_csv(os.path.join(self.logs_dir, report_fname), index=True)

        # Confusion matrix (combined)
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(self.class_names)))
        self.utils.plot_confusion_matrix(cm, self.class_names, model_name, fold, epoch)

        # Species-level confusion
        self.utils.plot_species_confusion_matrix(cm, self.class_names, model_name, fold, epoch)

        # Misclassified examples
        self.utils.plot_misclassified_examples(
            self.model,
            loader,
            self.class_names,
            model_name,
            fold,
            epoch,
            device=self.device
        )

        # Return summary metrics
        return avg_val_loss, accuracy, report_dict, df_media, summary_df
