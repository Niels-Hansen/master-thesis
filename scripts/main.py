import argparse
import torch
from data import DataLoaderFactory
from models import ModelFactory
from evaluator import Evaluator
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', help='Run setup without training')
parser.add_argument('--source-dir', type=str, default="/work3/s233780/ResizedData", help='Path to source dataset')
parser.add_argument('--k-folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--model-name', type=str, default="efficientnet_v2_s", help='Name of the model to use (e.g., efficientnet_v2_s, vit_b_16, resnext101_32x8d)')

# Pass the model to the trainer
args = parser.parse_args()

model_name = args.model_name
source_dir = args.source_dir
num_epochs = 10
k_folds = args.k_folds
mapping_file = 'DataDescription.xlsx'  # Path to the mapping file
temp_dir='../tempdata/kfold'

data_loader_factory = DataLoaderFactory(source_dir,mapping_file,temp_dir)
full_dataset = data_loader_factory.prepare_data()

print("Dataset classes:", full_dataset.classes)

num_classes = len(full_dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the model using the selected model name
model_factory = ModelFactory(num_classes, device)
model = model_factory.get_model(model_name)

evaluator = Evaluator(device)
trainer = Trainer(model, evaluator, device, num_epochs)

if not args.dry_run:
    trainer.train(full_dataset, k_folds)
else:
    print("Dry run complete: dataset prepared, models initialized.")