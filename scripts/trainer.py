import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time

class Trainer:
    def __init__(self, model, evaluator, device, num_epochs):
        self.model = model  # Single model passed to the trainer
        self.evaluator = evaluator
        self.device = device
        self.num_epochs = num_epochs

    def train(self, full_dataset, k_folds):
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        start_time = time.time()

        all_fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
            print(f"Fold {fold + 1}/{k_folds}")

            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            writer = SummaryWriter(log_dir=f"runs/{self.model.__class__.__name__}_fold{fold + 1}")
            history = {'loss': [], 'val_acc': []}

            for epoch in range(self.num_epochs):
                print(f"  Epoch {epoch + 1}/{self.num_epochs}")
                self.model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                avg_loss = running_loss / len(train_loader)
                val_acc, val_report = self.evaluator.evaluate(self.model, val_loader, full_dataset.classes, model_name=self.model.__class__.__name__, fold=fold+1, epoch=epoch+1)
                history['loss'].append(avg_loss)
                history['val_acc'].append(val_acc)
                print(f"    Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")
                print(val_report)

                writer.add_scalar('Loss/train', avg_loss, epoch + 1)
                writer.add_scalar('Accuracy/val', val_acc, epoch + 1)

            # Record final val accuracy for the fold
            all_fold_accuracies.append(history['val_acc'][-1])

            # Plot learning curves
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, self.num_epochs+1), history['loss'], label='Training Loss')
            plt.plot(range(1, self.num_epochs+1), history['val_acc'], label='Validation Accuracy')
            plt.title(f"Learning Curve - Fold {fold+1}")
            plt.xlabel("Epoch")
            plt.ylabel("Metric")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"visualizations/learning_curve_fold{fold+1}.png")
            plt.close()

            torch.save(self.model.state_dict(), f"{self.model.__class__.__name__}_fold{fold + 1}.pth")
            writer.close()
            print(f"Model for fold {fold + 1} saved.")

        # Log average accuracy across folds
        avg_acc = sum(all_fold_accuracies) / len(all_fold_accuracies)
        print(f"\nAverage Validation Accuracy Across Folds: {avg_acc:.4f}")

        end_time = time.time()
        print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")