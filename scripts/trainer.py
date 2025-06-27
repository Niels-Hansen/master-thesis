import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import time
import csv
import pandas as pd
import os
from utils import Utils
import re

class Trainer:
    def __init__(self, model_factory, evaluator, device, num_epochs, train_transform, val_transform):
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.device = device
        self.num_epochs = num_epochs
        self.utils = Utils()
        self.train_transform = train_transform
        self.val_transform   = val_transform
        os.makedirs("logs", exist_ok=True)
        # Epoch-level metrics:
        self.metrics_file = "logs/metrics.csv"
        with open(self.metrics_file, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Model", "Fold", "Epoch", "Accuracy", "Loss", "Val Loss", "Val Accuracy"])

        # Per-class reports:
        self.classif_file = "logs/classification_report.csv"
        with open(self.classif_file, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Model", "Fold", "Epoch",
                                 "Class", "Precision", "Recall", "F1-Score", "Support"])

    def train(self, full_dataset, k_folds, model_name):
        #kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        labels = [label for _, label in full_dataset.imgs]  # For stratification
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        
        
        start_time = time.time()

        all_fold_accuracies = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset,labels)):
            print(f"Fold {fold + 1}/{k_folds}")
            model = self.model_factory.get_model(model_name).to(self.device) # Move model to the appropriate device

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.002)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
            # Create DataLoaders for the current fold
            #train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=64, shuffle=True)
            #val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=64, shuffle=False)
            #full_dataset.transform = full_dataset.loader_factory.train_transforms
            full_dataset.transform = self.train_transform
            train_loader = DataLoader(
                Subset(full_dataset, train_idx),
                batch_size=32,
                shuffle=True,
            )
            #full_dataset.transform = full_dataset.loader_factory.val_transforms
            full_dataset.transform = self.val_transform
            val_loader = DataLoader(
                Subset(full_dataset, val_idx),
                batch_size=32,
                shuffle=False,
            )

            self.evaluator.model     = model
            self.evaluator.criterion = criterion

            self.tb_writer = SummaryWriter(log_dir=f"runs/{model_name}_fold{fold + 1}")
            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            for epoch in range(self.num_epochs):
                print(f"  Epoch {epoch + 1}/{self.num_epochs}")
                model.train()
                running_loss = 0.0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                scheduler.step()  # Step the learning rate scheduler

                avg_loss = running_loss / len(train_loader)
                avg_val_loss, val_acc, report_dict, df_media, summary_df = \
                self.evaluator.evaluate(
                    val_loader,
                    model_name=model_name,
                    fold=fold+1,
                    epoch=epoch+1
                )

                correct = sum(
                    (model(inputs.to(self.device)).argmax(1) == labels.to(self.device)).sum().item()
                    for inputs, labels in train_loader
                )
                train_acc = correct / len(train_loader.dataset)
                history['train_acc'].append(train_acc)
                history['train_loss'].append(avg_loss)
                history['val_acc'].append(val_acc)
                history['val_loss'].append(avg_val_loss)
                print(f"    Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")
                print(report_dict)

                self.tb_writer.add_scalar('Loss/train', avg_loss, epoch + 1)
                self.tb_writer.add_scalar('Loss/val', avg_val_loss, epoch + 1)
                self.tb_writer.add_scalar('Accuracy/val', val_acc, epoch + 1)

                with open(self.metrics_file, "a", newline="") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([
                        model.__class__.__name__,
                        fold + 1,
                        epoch,
                        f"{train_acc:.4f}",            # Accuracy
                        f"{avg_loss:.4f}",             # Loss
                        f"{avg_val_loss:.4f}",         # Val Loss
                        f"{val_acc:.4f}"                   # Val Accuracy
                    ])

                df = pd.DataFrame(report_dict).T.reset_index().rename(columns={
                    "index": "Class",
                    "precision": "Precision",
                    "recall":    "Recall",
                    "f1-score":  "F1-Score",
                    "support":   "Support"
                })
                df["Model"] = model_name
                df["Fold"]  = fold + 1
                df["Epoch"] = epoch
                df = df[["Model","Fold","Epoch","Class","Precision","Recall","F1-Score","Support"]]
                
                df.to_csv(self.classif_file, mode="a", header=False, index=False)
            checkpoint_dir = f"checkpoints_{model_name}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
            model.state_dict(),
            os.path.join(checkpoint_dir, f"fold_{fold+1}.pt")
            )

            self.utils.plot_learning_curves(    
                history,
                model_name,
                fold + 1
            )
                

            # Record final val accuracy for the fold
            all_fold_accuracies.append(history['val_acc'][-1])

            # Plot learning curves
            plt.figure(figsize=(20, 20))
            plt.plot(range(1, self.num_epochs+1), history['train_loss'], label='Training Loss')
            plt.plot(range(1, self.num_epochs+1), history['val_loss'], label='Validation Loss')
            plt.plot(range(1, self.num_epochs+1), history['train_acc'], label='Training Accuracy')
            plt.plot(range(1, self.num_epochs+1), history['val_acc'], label='Validation Accuracy')
            plt.title(f"Learning Curve - Fold {fold+1}")
            plt.xlabel("Epoch")
            plt.ylabel("Metric")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"visualizations/{model_name}_learning_curve_fold{fold+1}.png")
            plt.close()

            self.tb_writer.close()
            print(f"Model for fold {fold + 1} saved.")

        # Log average accuracy across folds
        avg_acc = sum(all_fold_accuracies) / len(all_fold_accuracies)
        print(f"\nAverage Validation Accuracy Across Folds: {avg_acc:.4f}")

        end_time = time.time()
        print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")