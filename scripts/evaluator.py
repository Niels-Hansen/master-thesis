import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

class Evaluator:
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, val_loader, class_names, model_name="model", fold=0, epoch=0):
        # switch model to evaluation mode
        model.eval()
        all_preds, all_labels = [], []

        # collect predictions and true labels
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # identify which labels appeared in this fold
        unique_labels = sorted(set(all_labels))
        # map these labels to their human-readable names
        filtered_class_names = [class_names[i] for i in unique_labels]

        # compute accuracy and classification report with explicit labels
        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels,
            all_preds,
            labels=unique_labels,
            target_names=filtered_class_names,
            zero_division=0
        )
        # compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # plot confusion matrix using matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"Confusion Matrix - {model_name} Fold {fold} Epoch {epoch}")
        plt.colorbar()
        tick_marks = range(len(unique_labels))
        plt.xticks(tick_marks, filtered_class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, filtered_class_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig(f"visualizations/confusion_matrix_{model_name}_fold{fold}_epoch{epoch}.png")
        plt.close()

        return acc, report
