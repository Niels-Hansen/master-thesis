import argparse
import torch
from data import DataLoaderFactory
from models import ModelFactory
from evaluator import Evaluator
from trainer import Trainer
from utils import Utils
from torchvision.datasets import ImageFolder

print("Starting training script...")

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', help='Run setup without training')
parser.add_argument('--source-dir', type=str, default="/work3/s233780/CircularMaskedData", help='Path to source dataset')
parser.add_argument('--k-folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--model-name', type=str, default="efficientnet_v2_s", help='Name of the model to use (e.g., efficientnet_v2_s, vit_b_16, resnext101_32x8d)')

print("Available models: efficientnet_v2_s, vit_b_16, resnext101_32x8d")
# Pass the model to the trainer
args = parser.parse_args()

model_name = args.model_name
source_dir = args.source_dir
num_epochs = 10
k_folds = args.k_folds
mapping_file = 'imageAnalysis_info.xlsx'  # Path to the mapping file
temp_dir='/work3/s233780/circledata/kfold'
print(f"Using source directory: {source_dir}")
#data_loader_factory = DataLoaderFactory(source_dir,mapping_file,temp_dir)
#full_dataset = data_loader_factory.prepare_data(args.model_name)
data_loader_factory = DataLoaderFactory(source_dir, mapping_file, temp_dir)
# this now just returns temp_dir
print("Preparing data...")
data_folder = data_loader_factory.prepare_data(args.model_name)
 # build a "bare" dataset — NO transform here
print("Loading dataset from:", data_folder)
full_dataset = ImageFolder(data_folder)
class_names  = full_dataset.classes
print("Dataset classes:", class_names)

# grab the two pipelines
train_tfms = data_loader_factory.train_transforms
val_tfms   = data_loader_factory.val_transforms

class_names = full_dataset.classes
print("Dataset classes:", full_dataset.classes)

num_classes = len(full_dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model using the selected model name
model_factory = ModelFactory(num_classes, device)

utils = Utils(base_dir="visualizations")
    # instantiate evaluator with all required args
evaluator = Evaluator(
    model=None,                   # placeholder, will set per-fold
    device=device,
    criterion=torch.nn.CrossEntropyLoss(),
    class_names=class_names,
    utils=utils,
    logs_dir="logs"
)
trainer = Trainer(model_factory, evaluator, device, num_epochs, train_transform=train_tfms, val_transform=val_tfms)

if not args.dry_run:
    trainer.train(full_dataset, k_folds, model_name)
else:
    print("Dry run complete: dataset prepared, models initialized.")