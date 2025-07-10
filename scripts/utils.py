import os
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix as sk_cm


class Utils:
    """
    Utility class for generating and saving visualization artifacts such as
    confusion matrices and misclassified example galleries.
    """
    def __init__(self, base_dir="visualizations", mean=None, std=None):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        # Default ImageNet normalization statistics if not provided
        self.mean = mean or torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = std  or torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def _ensure_model_dir(self, model_name):
        model_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def plot_confusion_matrix(self, cm, class_names, model_name, fold, epoch, suffix=""):
        """
        Plots and saves a confusion matrix heatmap.
        """
        num_classes = len(class_names)
        size = max(10, num_classes * 0.25)
        plt.figure(figsize=(size, size))
        sns.heatmap(
            cm,
            annot=False,
            fmt="d",
            cmap="viridis",
            cbar=True,
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            linecolor='lightgrey'
        )
        plt.title(f"Confusion Matrix - {model_name} Fold {fold} Epoch {epoch} {suffix}")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        font_size = max(2, 10 - num_classes // 15)
        plt.xticks(rotation=90, ha='right', fontsize=font_size)
        plt.yticks(rotation=0, fontsize=font_size)
        plt.tight_layout()
        model_dir = self._ensure_model_dir(model_name)
        fname = f"confusion_matrix_{model_name}_fold{fold}_epoch{epoch}{suffix}.png"
        plt.savefig(os.path.join(model_dir, fname))
        plt.close()

    def plot_species_confusion_matrix(self, cm_combined, full_class_names, model_name, fold, epoch):

        species_names_raw = []
        for name in full_class_names:
            match = re.match(r'(.+?)_[A-Za-z]+$', name)
            species_names_raw.append(match.group(1) if match else name)
        unique_species = sorted(set(species_names_raw))
        mapping = {s: i for i, s in enumerate(unique_species)}

        species_cm = sk_cm(
            [mapping[species_names_raw[i]] for i in range(len(species_names_raw))],
            [mapping[species_names_raw[i]] for i in range(len(species_names_raw))],
            labels=list(range(len(unique_species)))
        )
        self.plot_confusion_matrix(
            species_cm,
            unique_species,
            model_name,
            fold,
            epoch,
            suffix="_species"
        )

    def plot_misclassified_examples(
        self,
        model,
        loader,
        class_names,
        model_name,
        fold,
        epoch,
        num_examples=5,
        device="cpu"
    ):
        model.eval()
        images, preds, labels = [], [], []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                batch_preds = outputs.argmax(dim=1)
                wrong = (batch_preds != targets).nonzero(as_tuple=True)[0]
                for idx in wrong:
                    if len(images) >= num_examples:
                        break
                    images.append(inputs[idx].cpu())
                    preds.append(batch_preds[idx].item())
                    labels.append(targets[idx].item())
                if len(images) >= num_examples:
                    break
        if not images:
            print(f"No misclassified examples for {model_name} fold {fold} epoch {epoch}")
            return

        fig, axes = plt.subplots(1, len(images), figsize=(4*len(images), 4))
        if len(images) == 1:
            axes = [axes]
        for ax, img, p, t in zip(axes, images, preds, labels):
            img = img * self.std + self.mean
            img = torch.clamp(img, 0, 1)
            ax.imshow(transforms.ToPILImage()(img))
            ax.set_title(f"True: {class_names[t]}\nPred: {class_names[p]}", color='red', fontsize=8)
            ax.axis('off')

        plt.suptitle(f"Misclassified - {model_name} Fold {fold} Epoch {epoch}")
        plt.tight_layout(rect=[0,0.03,1,0.95])
        model_dir = self._ensure_model_dir(model_name)
        fname = f"misclassified_{model_name}_fold{fold}_epoch{epoch}.png"
        plt.savefig(os.path.join(model_dir, fname))
        plt.close()

    def plot_learning_curves(self, history, model_name, fold):

        epochs = range(1, len(history['train_loss']) + 1)
        # Loss curves
        plt.figure()
        plt.plot(epochs, history['train_loss'], label='Train Loss')
        plt.plot(epochs, history['val_loss'],   label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Fold {fold} Loss Curves - {model_name}')
        plt.tight_layout()
        loss_fname = os.path.join(self._ensure_model_dir(model_name), f'learning_loss_{model_name}_fold{fold}.png')
        plt.savefig(loss_fname)
        plt.close()
        # Accuracy curves
        plt.figure()
        plt.plot(epochs, history['train_acc'], label='Train Acc')
        plt.plot(epochs, history['val_acc'],   label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'Fold {fold} Accuracy Curves - {model_name}')
        plt.tight_layout()
        acc_fname = os.path.join(self._ensure_model_dir(model_name), f'learning_acc_{model_name}_fold{fold}.png')
        plt.savefig(acc_fname)
        plt.close()
        
    def cross_val_report(self, all_true, all_pred, class_names, logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

        cm = confusion_matrix(all_true, all_pred, labels=list(range(len(class_names))))

        report = classification_report(
            all_true,
            all_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0
        )

        report_path = os.path.join(logs_dir, "cross_val_report.txt")
        with open(report_path, "w") as f:
            f.write("Cross-validated Confusion Matrix")
            f.write(str(cm) + "\n\n")
            f.write("Cross-validated Classification Report")
            f.write(report)