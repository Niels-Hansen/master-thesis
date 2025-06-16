import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for better heatmaps
import numpy as np # Import numpy for array operations
import os
import re # Import regex for parsing species from class names
from torchvision import transforms # Needed for image denormalization

class Evaluator:
    def __init__(self, device):
        self.device = device
        os.makedirs("visualizations", exist_ok=True)

    def _plot_confusion_matrix(self, cm, class_names_to_plot, model_name, fold, epoch, fig_name_suffix=""):
        """
        Helper to plot and save a confusion matrix using Seaborn.
        Adjusts figure size and font size for readability with many classes.
        """
        num_classes_to_plot = len(class_names_to_plot) # Use the specific class_names list for this plot
        
        fig_size_base = 10
        scaling_factor = 0.25
        fig_size = max(fig_size_base, num_classes_to_plot * scaling_factor)
        
        plt.figure(figsize=(fig_size, fig_size))
        
        sns.heatmap(cm, annot=False, fmt="d", cmap="viridis", cbar=True,
                    xticklabels=class_names_to_plot, yticklabels=class_names_to_plot, # Use specific class_names
                    linewidths=.5, linecolor='lightgrey')
        
        plt.title(f"Confusion Matrix - {model_name} Fold {fold} Epoch {epoch} {fig_name_suffix}")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        
        tick_font_size = max(2, 10 - num_classes_to_plot // 15)
        plt.xticks(rotation=90, ha='right', fontsize=tick_font_size)
        plt.yticks(rotation=0, fontsize=tick_font_size)
        
        plt.tight_layout()
        
        model_viz_dir = os.path.join("visualizations", model_name)
        os.makedirs(model_viz_dir, exist_ok=True)
        
        plt.savefig(os.path.join(model_viz_dir, f"confusion_matrix_{model_name}_fold{fold}_epoch{epoch}{fig_name_suffix}.png"))
        plt.close()

    def _plot_misclassified_examples(self, model, loader, class_names, model_name, fold, epoch, num_examples=5):
        """Plots a few misclassified examples, showing true vs predicted labels."""
        model.eval()
        misclassified_images = []
        misclassified_preds = []
        misclassified_labels = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)

                incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]
                
                for idx in incorrect_indices:
                    if len(misclassified_images) < num_examples:
                        misclassified_images.append(inputs[idx].cpu())
                        misclassified_preds.append(preds[idx].item())
                        misclassified_labels.append(labels[idx].item())
                    else:
                        break
                if len(misclassified_images) >= num_examples:
                    break

        if misclassified_images:
            fig, axes = plt.subplots(1, len(misclassified_images), figsize=(4 * len(misclassified_images), 4))
            
            if len(misclassified_images) == 1:
                axes = [axes]
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            for i, ax in enumerate(axes):
                img_tensor = misclassified_images[i]
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)
                img_pil = transforms.ToPILImage()(img_tensor)
                
                ax.imshow(img_pil)
                ax.set_title(f"True: {class_names[misclassified_labels[i]]}\nPred: {class_names[misclassified_preds[i]]}",
                             color='red', fontsize=8)
                ax.axis('off')

            plt.suptitle(f"Misclassified Examples - {model_name} Fold {fold} Epoch {epoch}", fontsize=10)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            model_viz_dir = os.path.join("visualizations", model_name)
            os.makedirs(model_viz_dir, exist_ok=True)
            
            plt.savefig(os.path.join(model_viz_dir, f"misclassified_examples_{model_name}_fold{fold}_epoch{epoch}.png"))
            plt.close()
        else:
            print(f"  No misclassified examples found for {model_name} in Fold {fold} Epoch {epoch} to plot.")


    def evaluate(self, model, loader, full_class_names, model_name="", fold=0, epoch=0):
        """
        Evaluates the model and generates various plots.
        Generates two confusion matrices: one for combined labels and one for species-level.
        """
        model.eval()
        all_preds_idx, all_labels_idx = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds_idx.extend(preds.cpu().numpy())
                all_labels_idx.extend(labels.cpu().numpy())

        # --- Overall Accuracy and Classification Report (on combined classes) ---
        acc = accuracy_score(all_labels_idx, all_preds_idx)
        # Use full_class_names for target_names and all possible labels (0 to num_classes-1)
        report = classification_report(
            all_labels_idx,
            all_preds_idx,
            labels=range(len(full_class_names)), # <-- Pass all possible label indices
            target_names=full_class_names,       # <-- Pass all class names
            zero_division=0
        )
        

        # --- Plot Combined Species_Media Confusion Matrix ---
        # Compute confusion matrix with all possible labels
        cm_combined = confusion_matrix(all_labels_idx, all_preds_idx, labels=range(len(full_class_names)))
        self._plot_confusion_matrix(cm_combined, full_class_names, model_name, fold, epoch, fig_name_suffix="_combined")


        # --- Plot Species-Level Confusion Matrix (Recommended for Readability) ---
        # 1. Map combined class indices to species-only names/indices
        species_names_raw_mapped = []
        for class_name in full_class_names:
            # Regex to capture "Genus _species" from names like "Penicillago _kabunica_YES"
            # Updated regex: captures everything before the last underscore if it's followed by letters (media)
            match = re.match(r'(.+)_[A-Za-z]+$', class_name) 
            if match:
                species_names_raw_mapped.append(match.group(1)) # This gets "Penicillago _kabunica"
            else:
                # Fallback for names like "single plate data" or if format doesn't match
                species_names_raw_mapped.append(class_name) 
                print(f"Warning: Class name '{class_name}' did not match expected pattern for species mapping. Using full name as fallback for species level CM.")

        species_names_unique = sorted(list(set(species_names_raw_mapped)))
        species_to_idx_map = {name: i for i, name in enumerate(species_names_unique)}

        species_labels_mapped = [species_to_idx_map[species_names_raw_mapped[label_idx]] for label_idx in all_labels_idx]
        species_preds_mapped = [species_to_idx_map[species_names_raw_mapped[pred_idx]] for pred_idx in all_preds_idx]

        # 2. Compute species-level confusion matrix
        # Compute confusion matrix with all possible species-level labels
        cm_species = confusion_matrix(species_labels_mapped, species_preds_mapped, labels=range(len(species_names_unique)))
        self._plot_confusion_matrix(cm_species, species_names_unique, model_name, fold, epoch, fig_name_suffix="_species_level")

        # --- Plot Misclassified Examples ---
        self._plot_misclassified_examples(model, loader, full_class_names, model_name, fold, epoch)

        return acc, report