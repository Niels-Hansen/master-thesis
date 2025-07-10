# Automated fungal identification using Deep



## Setup

To train the models.

1. First clone the repo

```bash
  git clone https://github.com/Niels-Hansen/master-thesis.git
```

2. Create and activate a virtualenv

```bash
python -m venv .venv

source .venv/bin/activate      # bash 

.venv\Scripts\activate         # PowerShell
```

3. Install dependencies
```bash
pip install --upgrade pip
pip install -r scripts/requirements.txt
pip install -r yolo/requirements.txt
```

## Data preparation

Insert the data with subfolders named by IBT numbers.
```bash
<SOURCE_DIR>/
├─ IBT_32047/
│  ├─ IBT_32047_144_CYA_0min.jpg
│  └─ ...
└─ IBT_32196/
```

## Training CNNs and ViT

```bash
python main.py --source-dir="/path/to/SOURCE_DIR" --model-name="efficientnet_v2_s"
  --k-folds=5
```

## Training YOLO
Remember to update the source directory
```bash
source_images_dir = "/work3/s233780/CircularMaskedData"
```
