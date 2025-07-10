import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
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
        self.logs_dir = "logs/cv"
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
        labels = [label for _, label in full_dataset.imgs] 
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=10)

    
        start_time = time.time()
        all_true = []
        all_pred = []
        all_fold_accuracies = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset,labels)):
            print(f"Fold {fold + 1}/{k_folds}")
            model = self.model_factory.get_model(model_name).to(self.device) 

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)  
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) 
                       
            train_ds = Subset(
                ImageFolder(full_dataset.root, transform=self.train_transform),
                train_idx
            )
            train_loader = DataLoader(
                train_ds, batch_size=128, shuffle=True,
                num_workers=4, pin_memory=True, persistent_workers=True
            )

            val_ds = Subset(
                ImageFolder(full_dataset.root, transform=self.val_transform),
                val_idx
            )
            val_loader = DataLoader(
                val_ds, batch_size=128, shuffle=False,
                num_workers=4, pin_memory=True, persistent_workers=True
            )

            self.evaluator.model     = model
            self.evaluator.criterion = criterion

            self.tb_writer = SummaryWriter(log_dir=f"runs/{model_name}_fold{fold + 1}")
            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            for epoch in range(self.num_epochs):
                print(f"  Epoch {epoch + 1}/{self.num_epochs}")
                model.train()
                running_loss = 0.0
                running_correct = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    running_correct += (outputs.argmax(1) == labels.to(self.device)).sum().item()
                    
                scheduler.step()  # Step the learning rate scheduler

                avg_loss = running_loss / len(train_loader)
                avg_val_loss, val_acc, report_dict, df_media, summary_df = \
                self.evaluator.evaluate(
                    val_loader,
                    model_name=model_name,
                    fold=fold+1,
                    epoch=epoch+1
                )
                
                all_true.extend(df_media['true'].tolist())
                all_pred.extend(df_media['pred'].tolist())

                train_acc = running_correct / len(train_loader.dataset)
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
                        f"{train_acc:.4f}",           
                        f"{avg_loss:.4f}",             
                        f"{avg_val_loss:.4f}",        
                        f"{val_acc:.4f}"                  
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
                epoch_dir = f"/work3/s233780/checkpoints_{model_name}"
                os.makedirs(epoch_dir, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        epoch_dir,
                        f"{model_name}_fold{fold+1}_epoch{epoch+1}.pt"
                    )
               )
                
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
            self.utils.cross_val_report(
                all_true,
                all_pred,
                self.evaluator.class_names,
                self.logs_dir
            )

            all_fold_accuracies.append(history['val_acc'][-1])

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

        avg_acc = sum(all_fold_accuracies) / len(all_fold_accuracies)
        print(f"\nAverage Validation Accuracy Across Folds: {avg_acc:.4f}")

        end_time = time.time()
        print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")