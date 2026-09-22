# dual-view-hate-speech-detection

## Setup
pip install -r requirements.txt
python -m spacy download en_core_web_sm

## Data
This project uses HateXplain (https://github.com/hate-alert/HateXplain) and
ImplicitHate (https://huggingface.co/datasets/SALT-NLP/ImplicitHate).
See data/README.md for download and preprocessing instructions.
Datasets are not redistributed here due to licensing.

## Training
python src/train_proposed.py --config configs/config.yaml
python src/train_baselines.py --model distilbert

## Evaluation
python src/evaluate.py --checkpoint results/proposed_best.pt

## Results
[Table 5/6 summary, or a link to the paper]

## Citation
[bibtex once accepted]
